//! Python bindings for `blip25-mbe`.
//!
//! Exposes the reference-shaped façade — `Rate`, `Vocoder`, `Transcoder`,
//! `LiveEncoder`, `LiveDecoder` — with zero-copy numpy interop on the
//! PCM boundary. As of blip25-mbe 0.3.0 the codec is a pure
//! bits-in/bits-out function with no runtime levers; the only remaining
//! encoder toggles are tone detection and the post-decode enhancement mode.

use blip25_mbe::enhancement::{ClassicalConfig, EnhancementMode};
use blip25_mbe::rate33::{frame as r33f, priority as r33p};
use blip25_mbe::reference_soft_decision as sd;
use blip25_mbe::vocoder::{self as bv, AnalysisOutputKind, Rate as BRate};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

/// Map a `VocoderError` to a Python `ValueError` with the upstream
/// `Display` text — preserves the actionable message without
/// inventing a new exception hierarchy.
fn map_err(e: bv::VocoderError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Map a `SoftDecisionError` to a Python `ValueError`, same policy as
/// [`map_err`].
fn map_sd_err(e: sd::SoftDecisionError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// On-wire rate / codec selection. The four variants cover the
/// production P25 storage and transport formats. See the upstream
/// `docs/wire_formats_and_storage.md` for the trade-offs.
#[pyclass(eq, eq_int, frozen, name = "Rate")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRate {
    /// P25 Phase 1 FDMA full-rate IMBE (18-byte FEC frame, 7 200 bps).
    #[pyo3(name = "IMBE_7200X4400")]
    Imbe7200x4400,
    /// IMBE info-only (11-byte frame, 4 400 bps).
    #[pyo3(name = "IMBE_4400X4400")]
    Imbe4400x4400,
    /// P25 Phase 2 TDMA AMBE+2 (9-byte FEC frame, 3 600 bps).
    #[pyo3(name = "AMBEPLUS2_3600X2450")]
    AmbePlus2_3600x2450,
    /// AMBE+2 info-only (7-byte frame, 2 450 bps). The 49 info bits are
    /// packed in **r34 column-interleave** order — byte-exact with the
    /// reference rate-index 34 no-FEC stream, *not* naive MSB-first
    /// sequential. Consumers needing natural / "AMBE_d" order (mbelib,
    /// IDAS/NXDN over-the-air) must de-interleave first.
    #[pyo3(name = "AMBEPLUS2_2450X2450")]
    AmbePlus2_2450x2450,
}

impl From<PyRate> for BRate {
    fn from(r: PyRate) -> Self {
        match r {
            PyRate::Imbe7200x4400 => BRate::Imbe7200x4400,
            PyRate::Imbe4400x4400 => BRate::Imbe4400x4400,
            PyRate::AmbePlus2_3600x2450 => BRate::AmbePlus2_3600x2450,
            PyRate::AmbePlus2_2450x2450 => BRate::AmbePlus2_2450x2450,
        }
    }
}

impl From<BRate> for PyRate {
    fn from(r: BRate) -> Self {
        // `Rate` is `#[non_exhaustive]` upstream; the catch-all
        // covers future variants when this crate is rebuilt against
        // a newer `blip25-mbe`. Currently unreachable.
        #[allow(unreachable_patterns)]
        match r {
            BRate::Imbe7200x4400 => PyRate::Imbe7200x4400,
            BRate::Imbe4400x4400 => PyRate::Imbe4400x4400,
            BRate::AmbePlus2_3600x2450 => PyRate::AmbePlus2_3600x2450,
            BRate::AmbePlus2_2450x2450 => PyRate::AmbePlus2_2450x2450,
            other => panic!("blip25-mbe-py: unmapped upstream Rate variant {other:?}"),
        }
    }
}

/// Post-decoder PCM enhancement mode. `NONE` is spec-faithful (no
/// post-processing) and is the upstream library default — the shipped
/// decode path is the reference console codec unaltered. `CLASSICAL`
/// applies the default biquad + peaking-EQ + output-gain chain (no
/// compressor, fade enabled) as an opt-in research post-filter.
///
/// Fine-grained tuning of the classical chain (custom biquads,
/// compressor knobs, fade samples) isn't exposed from Python yet —
/// open an issue if you need it.
#[pyclass(eq, eq_int, frozen, name = "EnhancementMode")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyEnhancementMode {
    #[pyo3(name = "NONE")]
    NoneMode,
    #[pyo3(name = "CLASSICAL")]
    Classical,
}

impl From<PyEnhancementMode> for EnhancementMode {
    fn from(m: PyEnhancementMode) -> Self {
        match m {
            PyEnhancementMode::NoneMode => EnhancementMode::None,
            PyEnhancementMode::Classical => EnhancementMode::Classical(ClassicalConfig::default()),
        }
    }
}

impl From<&EnhancementMode> for PyEnhancementMode {
    fn from(m: &EnhancementMode) -> Self {
        match m {
            EnhancementMode::None => PyEnhancementMode::NoneMode,
            EnhancementMode::Classical(_) => PyEnhancementMode::Classical,
        }
    }
}

fn output_kind_str(k: AnalysisOutputKind) -> &'static str {
    // The catch-all guards against future variants when this crate is
    // rebuilt against a newer `blip25-mbe`.
    #[allow(unreachable_patterns)]
    match k {
        AnalysisOutputKind::Voice => "voice",
        AnalysisOutputKind::Silence => "silence",
        AnalysisOutputKind::Tone { .. } => "tone",
        _ => "unknown",
    }
}

/// One vocoder channel. Stateful across frames (analysis history,
/// decoder cross-frame memory). Use [`Vocoder.reset`] to clear between
/// independent streams.
#[pyclass(name = "Vocoder", unsendable)]
pub struct PyVocoder {
    inner: bv::Vocoder,
}

#[pymethods]
impl PyVocoder {
    #[new]
    fn new(rate: PyRate) -> Self {
        Self { inner: bv::Vocoder::new(rate.into()) }
    }

    /// The rate this vocoder was constructed for.
    #[getter]
    fn rate(&self) -> PyRate {
        self.inner.rate().into()
    }

    /// PCM samples in one frame (`160` for all current rates).
    #[getter]
    fn frame_samples(&self) -> usize {
        self.inner.frame_samples()
    }

    /// On-wire byte count for one frame at this rate.
    #[getter]
    fn fec_frame_bytes(&self) -> usize {
        self.inner.fec_frame_bytes()
    }

    /// Encode one PCM frame to its on-wire bytes. `pcm` must be a
    /// 1-D `np.int16` array of length [`frame_samples`].
    fn encode_pcm<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArray1<'py, i16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.encode_pcm(pcm.as_slice()?).map_err(map_err)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Decode one on-wire frame back to PCM. `bits` must be exactly
    /// [`fec_frame_bytes`] bytes long.
    fn decode_bits<'py>(
        &mut self,
        py: Python<'py>,
        bits: &[u8],
    ) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let pcm = self.inner.decode_bits(bits).map_err(map_err)?;
        Ok(pcm.into_pyarray_bound(py))
    }

    /// Clear cross-frame state. Call between independent streams /
    /// callers sharing one vocoder instance.
    fn reset(&mut self) {
        self.inner.reset();
    }

    // ── encoder toggles ─────────────────────────────────────────
    // blip25-mbe 0.3.0 is a pure bits-in/bits-out codec; the only two
    // remaining runtime toggles are encode-side tone detection and the
    // post-decode enhancement mode.

    #[getter]
    fn tone_detection(&self) -> bool { self.inner.tone_detection() }

    fn set_tone_detection(&mut self, on: bool) { self.inner.set_tone_detection(on); }

    #[getter]
    fn enhancement(&self) -> PyEnhancementMode { self.inner.enhancement().into() }

    fn set_enhancement(&mut self, mode: PyEnhancementMode) {
        self.inner.set_enhancement(mode.into());
    }

    // ── diagnostics ────────────────────────────────────────────

    /// `"voice"` / `"silence"` / `"tone"` for the last encoded frame,
    /// or `None` if no encode has run since the last [`reset`].
    fn last_output_kind(&self) -> Option<&'static str> {
        self.inner
            .last_stats()
            .analysis
            .as_ref()
            .map(|a| output_kind_str(a.output))
    }

    /// `(id, amplitude)` for the last frame's Annex T tone detection,
    /// or `None` if the last encoded frame was not a tone frame (tone
    /// detection disabled, no tone found, or no encode yet).
    fn last_tone_detection(&self) -> Option<(u8, u8)> {
        self.inner
            .last_stats()
            .analysis
            .as_ref()
            .and_then(|a| match a.output {
                AnalysisOutputKind::Tone { id, amplitude } => Some((id, amplitude)),
                _ => None,
            })
    }

    fn __repr__(&self) -> String {
        format!("Vocoder(rate=Rate.{:?})", self.inner.rate())
    }
}

/// Chunk-driven streaming encoder. Buffers caller-supplied PCM into
/// 160-sample frames and emits one wire-byte blob per completed
/// frame. Pass any chunk size you have; partial frames are held until
/// the next [`push`].
#[pyclass(name = "LiveEncoder", unsendable)]
pub struct PyLiveEncoder {
    inner: bv::LiveEncoder,
}

#[pymethods]
impl PyLiveEncoder {
    #[new]
    fn new(rate: PyRate) -> Self {
        Self { inner: bv::LiveEncoder::new(rate.into()) }
    }

    #[getter]
    fn rate(&self) -> PyRate {
        self.inner.rate().into()
    }

    #[getter]
    fn pending_samples(&self) -> usize {
        self.inner.pending_samples()
    }

    /// Push PCM. Returns a list of `bytes`, one per completed frame.
    /// Frames that fail to encode raise `ValueError` at the failing
    /// frame's slot.
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArray1<'py, i16>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let mut out = Vec::new();
        for r in self.inner.push(pcm.as_slice()?) {
            let bytes = r.map_err(map_err)?;
            out.push(PyBytes::new_bound(py, &bytes));
        }
        Ok(out)
    }

    /// Finish the stream: pad any pending residue with zeros and drain
    /// the codec's one-frame look-ahead. Returns a list of `bytes`, one
    /// per remaining frame (0, 1, or 2 frames — the look-ahead makes the
    /// tail unbounded to a single frame). Empty if nothing was pending.
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let frames = self.inner.flush().map_err(map_err)?;
        Ok(frames
            .iter()
            .map(|b| PyBytes::new_bound(py, b))
            .collect())
    }

    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Chunk-driven streaming decoder. Buffers caller-supplied wire
/// bytes into [`fec_frame_bytes`]-sized frames and emits PCM blocks
/// per decoded frame.
#[pyclass(name = "LiveDecoder", unsendable)]
pub struct PyLiveDecoder {
    inner: bv::LiveDecoder,
}

#[pymethods]
impl PyLiveDecoder {
    #[new]
    fn new(rate: PyRate) -> Self {
        Self { inner: bv::LiveDecoder::new(rate.into()) }
    }

    #[getter]
    fn rate(&self) -> PyRate {
        self.inner.rate().into()
    }

    #[getter]
    fn pending_bytes(&self) -> usize {
        self.inner.pending_bytes()
    }

    /// Push wire bytes. Returns a list of `np.int16` arrays, one per
    /// completed frame.
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        bits: &[u8],
    ) -> PyResult<Vec<Bound<'py, PyArray1<i16>>>> {
        let mut out = Vec::new();
        for r in self.inner.push(bits) {
            let pcm = r.map_err(map_err)?;
            out.push(pcm.into_pyarray_bound(py));
        }
        Ok(out)
    }

    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Wire-format bridge — decodes one rate and re-encodes to another.
/// Used for Phase 1 IMBE ⇄ Phase 2 AMBE+2 conversion and for same-
/// codec FEC ↔ no-FEC pairs.
#[pyclass(name = "Transcoder", unsendable)]
pub struct PyTranscoder {
    inner: bv::Transcoder,
}

#[pymethods]
impl PyTranscoder {
    #[new]
    fn new(from: PyRate, to: PyRate) -> PyResult<Self> {
        let inner = bv::Transcoder::new(from.into(), to.into()).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Transcode one input frame to the destination rate.
    fn transcode<'py>(
        &mut self,
        py: Python<'py>,
        bits: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let out = self.inner.transcode(bits).map_err(map_err)?;
        Ok(PyBytes::new_bound(py, &out))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ── reference soft-decision handoff ─────────────────────────────────
// The 4-bit soft-decision (LLR) packet format for soft-FEC interchange
// with AMBE-2000/2020/3000 hardware. Exposed as the
// `blip25_mbe.reference_soft_decision` submodule.

// The soft-decision packet SHAPE constants (header word, overhead/data
// word split, slot count) are fixed by the wire format and are internal
// (`pub(crate)`) to blip25-mbe 0.3.0. Mirror them here for the Python
// surface; they cannot change without a wire break.
const SD_HEADER: u16 = 0x13EC;
const SD_OVERHEAD_WORDS: usize = 12;
const SD_DATA_WORDS: usize = 48;
const SD_SLOTS: usize = 192;

/// The 12 overhead words of an AMBE-2000/2020 soft-decision packet,
/// minus the fixed `0x13EC` header. The five `rate_info` words and the
/// control words are reference-/rate-specific and caller-supplied.
#[pyclass(name = "SdPacketHeader")]
#[derive(Clone)]
pub struct PySdPacketHeader {
    inner: sd::SdPacketHeader,
}

#[pymethods]
impl PySdPacketHeader {
    /// Construct a header. The common case passes only `rate_info`
    /// (the five rate-control words); all other overhead fields default
    /// to zero. `rate_info` must be a sequence of exactly five `u16`.
    #[new]
    #[pyo3(signature = (
        rate_info = [0u16; 5],
        *,
        power_control = 0,
        control_word1 = 0,
        dtmf_control = 0,
        control_word2 = 0,
    ))]
    fn new(
        rate_info: [u16; 5],
        power_control: u8,
        control_word1: u8,
        dtmf_control: u16,
        control_word2: u16,
    ) -> Self {
        Self {
            inner: sd::SdPacketHeader {
                power_control,
                control_word1,
                rate_info,
                dtmf_control,
                control_word2,
            },
        }
    }

    #[getter]
    fn power_control(&self) -> u8 { self.inner.power_control }
    #[setter]
    fn set_power_control(&mut self, v: u8) { self.inner.power_control = v; }

    #[getter]
    fn control_word1(&self) -> u8 { self.inner.control_word1 }
    #[setter]
    fn set_control_word1(&mut self, v: u8) { self.inner.control_word1 = v; }

    #[getter]
    fn rate_info(&self) -> Vec<u16> { self.inner.rate_info.to_vec() }
    #[setter]
    fn set_rate_info(&mut self, v: [u16; 5]) { self.inner.rate_info = v; }

    #[getter]
    fn dtmf_control(&self) -> u16 { self.inner.dtmf_control }
    #[setter]
    fn set_dtmf_control(&mut self, v: u16) { self.inner.dtmf_control = v; }

    #[getter]
    fn control_word2(&self) -> u16 { self.inner.control_word2 }
    #[setter]
    fn set_control_word2(&mut self, v: u16) { self.inner.control_word2 = v; }

    fn __eq__(&self, other: &Self) -> bool { self.inner == other.inner }

    fn __repr__(&self) -> String {
        format!(
            "SdPacketHeader(rate_info={:?}, power_control={}, control_word1={}, dtmf_control={}, control_word2={})",
            self.inner.rate_info,
            self.inner.power_control,
            self.inner.control_word1,
            self.inner.dtmf_control,
            self.inner.control_word2,
        )
    }
}

/// Convert one native soft bit (`i8` LLR: sign = hard decision with
/// `> 0` meaning `1`, magnitude = confidence) into a reference 4-bit
/// soft-decision value in `0..=15` (offset-binary, MSB = hard bit).
#[pyfunction]
fn llr_to_sd_nibble(llr: i8) -> u8 { sd::llr_to_sd_nibble(llr) }

/// Inverse of [`llr_to_sd_nibble`]: map a reference 4-bit SD value back
/// to a representative `i8` LLR. Only the low nibble of `n` is used.
#[pyfunction]
fn sd_nibble_to_llr(n: u8) -> i8 { sd::sd_nibble_to_llr(n) }

/// Pack soft channel bits (one `int8` LLR per bit, `SD0` first) into a
/// 60-word reference soft-decision decoder packet (returned as a
/// `uint16` array). Fewer than 192 bits is normal; unused slots fill
/// with most-confident-0. Raises `ValueError` if more than 192 bits given.
#[pyfunction]
fn pack_channel_bits<'py>(
    py: Python<'py>,
    channel_llrs: PyReadonlyArray1<'py, i8>,
    header: &PySdPacketHeader,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let words =
        sd::pack_channel_bits(channel_llrs.as_slice()?, &header.inner).map_err(map_sd_err)?;
    Ok(words.to_vec().into_pyarray_bound(py))
}

/// Unpack a 60-word reference soft-decision packet (`uint16` array) into
/// its `(SdPacketHeader, int8[192] LLRs)`. Verifies the `0x13EC` header.
#[pyfunction]
fn unpack_packet<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'py, u16>,
) -> PyResult<(PySdPacketHeader, Bound<'py, PyArray1<i8>>)> {
    let slice = words.as_slice()?;
    let arr: &[u16; sd::SD_PACKET_WORDS] = slice.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "expected {} packet words, got {}",
            sd::SD_PACKET_WORDS,
            slice.len()
        ))
    })?;
    let (header, llrs) = sd::unpack_packet(arr).map_err(map_sd_err)?;
    Ok((PySdPacketHeader { inner: header }, llrs.to_vec().into_pyarray_bound(py)))
}

/// Serialize a 60-word packet (`uint16` array) to 120 big-endian bytes
/// (high byte of each word first — the host wire order).
#[pyfunction]
fn packet_to_bytes<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'py, u16>,
) -> PyResult<Bound<'py, PyBytes>> {
    let slice = words.as_slice()?;
    let arr: &[u16; sd::SD_PACKET_WORDS] = slice.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "expected {} packet words, got {}",
            sd::SD_PACKET_WORDS,
            slice.len()
        ))
    })?;
    Ok(PyBytes::new_bound(py, &sd::packet_to_bytes(arr)))
}

/// Pack soft channel bits (`int8` LLRs) into the raw USB-3000 SD nibble
/// stream (`*_sd.bit` format): two 4-bit values per byte, `SD0` in the
/// high nibble. No header or rate words.
#[pyfunction]
fn pack_nibble_stream<'py>(
    py: Python<'py>,
    channel_llrs: PyReadonlyArray1<'py, i8>,
) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new_bound(
        py,
        &sd::pack_nibble_stream(channel_llrs.as_slice()?),
    ))
}

/// Unpack a raw USB-3000 SD nibble stream (`bytes`) into `int8` LLRs,
/// `SD0` first. Inverse of [`pack_nibble_stream`].
#[pyfunction]
fn unpack_nibble_stream<'py>(py: Python<'py>, bytes: &[u8]) -> Bound<'py, PyArray1<i8>> {
    sd::unpack_nibble_stream(bytes).into_pyarray_bound(py)
}

/// Build and register the `reference_soft_decision` submodule.
fn register_reference_sd(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "reference_soft_decision")?;
    m.add_class::<PySdPacketHeader>()?;
    m.add_function(wrap_pyfunction!(llr_to_sd_nibble, &m)?)?;
    m.add_function(wrap_pyfunction!(sd_nibble_to_llr, &m)?)?;
    m.add_function(wrap_pyfunction!(pack_channel_bits, &m)?)?;
    m.add_function(wrap_pyfunction!(unpack_packet, &m)?)?;
    m.add_function(wrap_pyfunction!(packet_to_bytes, &m)?)?;
    m.add_function(wrap_pyfunction!(pack_nibble_stream, &m)?)?;
    m.add_function(wrap_pyfunction!(unpack_nibble_stream, &m)?)?;
    m.add("SD_HEADER", SD_HEADER)?;
    m.add("SD_PACKET_WORDS", sd::SD_PACKET_WORDS)?;
    m.add("SD_OVERHEAD_WORDS", SD_OVERHEAD_WORDS)?;
    m.add("SD_DATA_WORDS", SD_DATA_WORDS)?;
    m.add("SD_SLOTS", SD_SLOTS)?;
    m.add("RATE_33_CHANNEL_BITS", sd::RATE_33_CHANNEL_BITS)?;
    m.add("IMBE_FULL_RATE_CHANNEL_BITS", sd::IMBE_FULL_RATE_CHANNEL_BITS)?;
    m.add("P25_FULLRATE_FEC", sd::P25_FULLRATE_FEC.to_vec())?;
    m.add("P25_FULLRATE_NOFEC", sd::P25_FULLRATE_NOFEC.to_vec())?;
    parent.add_submodule(&m)?;
    // Register in sys.modules so
    // `import blip25_mbe._blip25_mbe.reference_soft_decision` resolves
    // (pyo3 submodules are not auto-registered).
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("blip25_mbe._blip25_mbe.reference_soft_decision", &m)?;
    Ok(())
}

// ── rate33 half-rate channel-frame toolkit ──────────────────────────
// Thin wrappers over `blip25_mbe::rate33::{frame, priority}` — the
// half-rate AMBE+2 *channel frame* layer that sits below the PCM↔wire
// façade. Lets callers work with the four info vectors `û₀..û₃`, the
// FEC code vectors `c₀..c₃`, the deprioritized `b̂₀..b̂₈` voice-parameter
// fields, and the r34 / natural (AMBE_d) no-FEC byte orders directly.
// Exposed as the `blip25_mbe.rate33` submodule.

/// Require an exact byte length, else a `ValueError` naming the caller.
fn require_len(bytes: &[u8], n: usize, what: &str) -> PyResult<()> {
    if bytes.len() != n {
        return Err(PyValueError::new_err(format!(
            "{what}: expected {n} bytes, got {}",
            bytes.len()
        )));
    }
    Ok(())
}

/// û₀ is a 12-bit info word — reject out-of-range seeds rather than
/// silently producing a garbage PN sequence (upstream only
/// `debug_assert`s this, which is a no-op in release builds).
fn check_u0(u0: u16) -> PyResult<()> {
    if u0 >= 4096 {
        return Err(PyValueError::new_err(format!(
            "û₀ must be a 12-bit value (0..=4095), got {u0}"
        )));
    }
    Ok(())
}

/// Unpack a packed MSB-first byte frame into 2-bit dibit values
/// (4 per byte). Used for the 9-byte FEC frame → 36 dibits handoff.
fn bytes_to_dibits(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 4);
    for &byte in bytes {
        out.push((byte >> 6) & 0b11);
        out.push((byte >> 4) & 0b11);
        out.push((byte >> 2) & 0b11);
        out.push(byte & 0b11);
    }
    out
}

/// Recover `û₀..û₃` from a 7-byte natural / AMBE_d (plain sequential
/// MSB-first) no-FEC frame. Distinct from the r34 column interleave.
fn natural_to_info_arr(bytes: &[u8]) -> [u16; 4] {
    let mut natural = [0u8; r33f::INFO_BITS_TOTAL as usize];
    for (i, slot) in natural.iter_mut().enumerate() {
        *slot = (bytes[i / 8] >> (7 - (i % 8))) & 1;
    }
    let mut info = [0u16; 4];
    let mut idx = 0usize;
    for (oi, &w) in r33f::INFO_WIDTHS.iter().enumerate() {
        let mut v = 0u16;
        for _ in 0..w {
            v = (v << 1) | u16::from(natural[idx]);
            idx += 1;
        }
        info[oi] = v;
    }
    info
}

/// Inverse of [`natural_to_info_arr`]: pack `û₀..û₃` into 7 natural
/// (AMBE_d, plain sequential) bytes. Bits 49..55 are zero pad.
fn info_to_natural_arr(info: &[u16; 4]) -> [u8; 7] {
    let mut out = [0u8; 7];
    let mut idx = 0usize;
    for (&w, &v) in r33f::INFO_WIDTHS.iter().zip(info.iter()) {
        for k in (0..w as usize).rev() {
            let bit = ((v >> k) & 1) as u8;
            out[idx / 8] |= bit << (7 - (idx % 8));
            idx += 1;
        }
    }
    out
}

/// Decoded 49-bit information layer of a half-rate (rate-33) frame: the
/// four info vectors `û₀..û₃` (LSB-aligned, widths per `INFO_WIDTHS`)
/// plus the per-vector FEC error counts.
#[pyclass(name = "Rate33Frame", frozen)]
#[derive(Clone)]
pub struct PyRate33Frame {
    inner: r33f::Frame,
}

#[pymethods]
impl PyRate33Frame {
    /// The four info vectors `û₀..û₃`.
    #[getter]
    fn info(&self) -> Vec<u16> {
        self.inner.info.to_vec()
    }

    /// Per-vector FEC error counts. `errors[0]` is 0–3 (or 255 for an
    /// uncorrectable extended-Golay `c₀`); `errors[1]` is 0–3; the
    /// uncoded `errors[2]`/`errors[3]` are always 0.
    #[getter]
    fn errors(&self) -> Vec<u8> {
        self.inner.errors.to_vec()
    }

    /// Total error count across all four vectors.
    fn error_total(&self) -> u16 {
        self.inner.error_total()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("Rate33Frame(info={:?}, errors={:?})", self.inner.info, self.inner.errors)
    }
}

// ── frame: no-FEC byte orders ───────────────────────────────────────

/// Pack the four info vectors `û₀..û₃` into a 7-byte **r34** no-FEC
/// frame (the `Rate.AMBEPLUS2_2450X2450` wire form), applying the
/// r34 3-way column interleave. Bits 49..55 are zero pad.
#[pyfunction]
fn pack_no_fec<'py>(py: Python<'py>, info: [u16; 4]) -> Bound<'py, PyBytes> {
    PyBytes::new_bound(py, &r33f::pack_no_fec(&info))
}

/// Inverse of [`pack_no_fec`]: recover `û₀..û₃` from a 7-byte r34
/// no-FEC frame (de-interleaves the r34 column order).
#[pyfunction]
fn unpack_no_fec(bytes: &[u8]) -> PyResult<Vec<u16>> {
    require_len(bytes, 7, "unpack_no_fec")?;
    Ok(r33f::unpack_no_fec(bytes).to_vec())
}

/// Recover `û₀..û₃` from a 7-byte **natural / AMBE_d** (plain
/// sequential MSB-first) no-FEC frame — the mbelib / IDAS / NXDN
/// over-the-air order, NOT the r34 column interleave.
#[pyfunction]
fn natural_to_info(bytes: &[u8]) -> PyResult<Vec<u16>> {
    require_len(bytes, 7, "natural_to_info")?;
    Ok(natural_to_info_arr(bytes).to_vec())
}

/// Inverse of [`natural_to_info`]: pack `û₀..û₃` into a 7-byte natural
/// (AMBE_d, plain sequential) no-FEC frame.
#[pyfunction]
fn info_to_natural<'py>(py: Python<'py>, info: [u16; 4]) -> Bound<'py, PyBytes> {
    PyBytes::new_bound(py, &info_to_natural_arr(&info))
}

// ── frame: FEC core (Annex-S interleave + Golay/PN) ─────────────────

/// Annex-S deinterleave 36 dibits → the 4 code vectors `c₀..c₃`.
#[pyfunction]
fn deinterleave(dibits: [u8; r33f::DIBITS_PER_FRAME]) -> Vec<u32> {
    r33f::deinterleave(&dibits).to_vec()
}

/// Annex-S interleave 4 code vectors `c₀..c₃` → 36 dibits. Inverse of
/// [`deinterleave`].
#[pyfunction]
fn interleave(codewords: [u32; 4]) -> Vec<u8> {
    r33f::interleave(&codewords).to_vec()
}

/// The half-rate PN sequence `p_r(0..=23)` seeded from `û₀`.
#[pyfunction]
fn pn_sequence(u0: u16) -> PyResult<Vec<u16>> {
    check_u0(u0)?;
    Ok(r33f::pn_sequence(u0).to_vec())
}

/// The 4 PN modulation masks `m̂₀..m̂₃` derived from `û₀`.
#[pyfunction]
fn modulation_masks(u0: u16) -> PyResult<Vec<u32>> {
    check_u0(u0)?;
    Ok(r33f::modulation_masks(u0).to_vec())
}

/// Demodulate the 4 code vectors against the `û₀`-seeded PN masks,
/// recovering `v̂₀..v̂₃`.
#[pyfunction]
fn demodulate(codewords: [u32; 4], u0: u16) -> PyResult<Vec<u32>> {
    check_u0(u0)?;
    Ok(r33f::demodulate(codewords, u0).to_vec())
}

/// Decode the 4 code vectors `c̃₀..c̃₃` (no interleave) through the
/// Golay/PN FEC core → [`Rate33Frame`]. The protocol-agnostic reuse
/// boundary for any half-rate AMBE+2 protocol (P25 Phase 2, DMR, NXDN).
#[pyfunction]
fn decode_code_vectors(codewords: [u32; 4]) -> PyRate33Frame {
    PyRate33Frame { inner: r33f::decode_code_vectors(codewords) }
}

/// Decode a half-rate frame from 36 P25-Phase-2 dibits → [`Rate33Frame`]
/// (Annex-S deinterleave + the Golay/PN FEC core).
#[pyfunction]
fn decode_frame(dibits: [u8; r33f::DIBITS_PER_FRAME]) -> PyRate33Frame {
    PyRate33Frame { inner: r33f::decode_frame(&dibits) }
}

/// Soft-decision decode a half-rate frame from 72 P25-Phase-2 soft bits
/// (`int8` LLRs: sign = hard bit, magnitude = confidence; layout
/// `[hi_dibit0, lo_dibit0, …]`) → [`Rate33Frame`].
#[pyfunction]
fn decode_frame_soft(soft: PyReadonlyArray1<'_, i8>) -> PyResult<PyRate33Frame> {
    let slice = soft.as_slice()?;
    let arr: &[i8; r33f::SOFT_BITS] = slice.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "decode_frame_soft: expected {} soft bits, got {}",
            r33f::SOFT_BITS,
            slice.len()
        ))
    })?;
    Ok(PyRate33Frame { inner: r33f::decode_frame_soft(arr) })
}

/// Encode 4 info vectors `û₀..û₃` into 72 air-interface bits (36
/// dibits). Inverse of [`decode_frame`].
#[pyfunction]
fn encode_frame(info: [u16; 4]) -> Vec<u8> {
    r33f::encode_frame(&info).to_vec()
}

// ── priority: info vectors ↔ b̂₀..b̂₈ voice-parameter fields ─────────

/// Prioritize the 9 quantized half-rate parameters `b̂₀..b̂₈` into the
/// 4 info vectors `û₀..û₃` (BABA-A §16.7).
#[pyfunction]
fn prioritize(b: [u16; r33p::AMBE_B_COUNT]) -> Vec<u16> {
    r33p::prioritize(&b).to_vec()
}

/// Deprioritize the 4 info vectors `û₀..û₃` into the 9 quantized
/// parameters `b̂₀..b̂₈`. Inverse of [`prioritize`].
#[pyfunction]
fn deprioritize(u: [u16; 4]) -> Vec<u16> {
    r33p::deprioritize(&u).to_vec()
}

// ── byte ↔ dibit helpers ────────────────────────────────────────────

/// Unpack a packed MSB-first byte string into 2-bit dibit values
/// (4 per byte). 9 FEC bytes → 36 dibits for [`decode_frame`].
#[pyfunction]
fn unpack_dibits(bytes: &[u8]) -> Vec<u8> {
    bytes_to_dibits(bytes)
}

/// Pack 2-bit dibit values back into MSB-first bytes (4 per byte).
/// Inverse of [`unpack_dibits`]; the dibit count must be a multiple of 4.
#[pyfunction]
fn pack_dibits<'py>(py: Python<'py>, dibits: Vec<u8>) -> PyResult<Bound<'py, PyBytes>> {
    if dibits.len() % 4 != 0 {
        return Err(PyValueError::new_err(format!(
            "pack_dibits: dibit count must be a multiple of 4, got {}",
            dibits.len()
        )));
    }
    let mut out = vec![0u8; dibits.len() / 4];
    for (i, &d) in dibits.iter().enumerate() {
        out[i / 4] |= (d & 0b11) << (6 - 2 * (i % 4));
    }
    Ok(PyBytes::new_bound(py, &out))
}

// ── field-dump conveniences (the whole pipeline in one call) ────────

/// Deprioritized `b̂₀..b̂₈` fields from a 9-byte FEC frame (the
/// `Rate.AMBEPLUS2_3600X2450` wire form): byte→dibit, Annex-S
/// deinterleave + Golay/PN FEC decode, then deprioritize.
#[pyfunction]
fn fields_from_fec(bytes: &[u8]) -> PyResult<Vec<u16>> {
    require_len(bytes, 9, "fields_from_fec")?;
    let dibits: [u8; r33f::DIBITS_PER_FRAME] = bytes_to_dibits(bytes)
        .try_into()
        .expect("9 bytes is exactly 36 dibits");
    Ok(r33p::deprioritize(&r33f::decode_frame(&dibits).info).to_vec())
}

/// Deprioritized `b̂₀..b̂₈` fields from a 7-byte **r34** no-FEC frame
/// (the `Rate.AMBEPLUS2_2450X2450` wire form).
#[pyfunction]
fn fields_from_no_fec(bytes: &[u8]) -> PyResult<Vec<u16>> {
    require_len(bytes, 7, "fields_from_no_fec")?;
    Ok(r33p::deprioritize(&r33f::unpack_no_fec(bytes)).to_vec())
}

/// Deprioritized `b̂₀..b̂₈` fields from a 7-byte **natural / AMBE_d**
/// (plain sequential) no-FEC frame — the mbelib / IDAS / NXDN order.
#[pyfunction]
fn fields_from_natural(bytes: &[u8]) -> PyResult<Vec<u16>> {
    require_len(bytes, 7, "fields_from_natural")?;
    Ok(r33p::deprioritize(&natural_to_info_arr(bytes)).to_vec())
}

/// Build and register the `rate33` submodule.
fn register_rate33(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "rate33")?;
    m.add_class::<PyRate33Frame>()?;
    m.add_function(wrap_pyfunction!(pack_no_fec, &m)?)?;
    m.add_function(wrap_pyfunction!(unpack_no_fec, &m)?)?;
    m.add_function(wrap_pyfunction!(natural_to_info, &m)?)?;
    m.add_function(wrap_pyfunction!(info_to_natural, &m)?)?;
    m.add_function(wrap_pyfunction!(deinterleave, &m)?)?;
    m.add_function(wrap_pyfunction!(interleave, &m)?)?;
    m.add_function(wrap_pyfunction!(pn_sequence, &m)?)?;
    m.add_function(wrap_pyfunction!(modulation_masks, &m)?)?;
    m.add_function(wrap_pyfunction!(demodulate, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_code_vectors, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_frame, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_frame_soft, &m)?)?;
    m.add_function(wrap_pyfunction!(encode_frame, &m)?)?;
    m.add_function(wrap_pyfunction!(prioritize, &m)?)?;
    m.add_function(wrap_pyfunction!(deprioritize, &m)?)?;
    m.add_function(wrap_pyfunction!(unpack_dibits, &m)?)?;
    m.add_function(wrap_pyfunction!(pack_dibits, &m)?)?;
    m.add_function(wrap_pyfunction!(fields_from_fec, &m)?)?;
    m.add_function(wrap_pyfunction!(fields_from_no_fec, &m)?)?;
    m.add_function(wrap_pyfunction!(fields_from_natural, &m)?)?;
    m.add("INFO_WIDTHS", r33f::INFO_WIDTHS.to_vec())?;
    m.add("CODE_WIDTHS", r33f::CODE_WIDTHS.to_vec())?;
    m.add("INFO_BITS_TOTAL", r33f::INFO_BITS_TOTAL)?;
    m.add("R34_BIT_ORDER", r33f::R34_BIT_ORDER.to_vec())?;
    m.add("PN_SEQ_LEN", r33f::PN_SEQ_LEN)?;
    m.add("DIBITS_PER_FRAME", r33f::DIBITS_PER_FRAME)?;
    m.add("SOFT_BITS", r33f::SOFT_BITS)?;
    m.add("AMBE_B_COUNT", r33p::AMBE_B_COUNT)?;
    m.add("AMBE_PARAM_WIDTHS", r33p::AMBE_PARAM_WIDTHS.to_vec())?;
    m.add("AMBE_VECTOR_WIDTHS", r33p::AMBE_VECTOR_WIDTHS.to_vec())?;
    parent.add_submodule(&m)?;
    // Register in sys.modules so `import blip25_mbe._blip25_mbe.rate33`
    // resolves (pyo3 submodules are not auto-registered).
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("blip25_mbe._blip25_mbe.rate33", &m)?;
    Ok(())
}

#[pymodule]
fn _blip25_mbe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRate>()?;
    m.add_class::<PyEnhancementMode>()?;
    m.add_class::<PyVocoder>()?;
    m.add_class::<PyLiveEncoder>()?;
    m.add_class::<PyLiveDecoder>()?;
    m.add_class::<PyTranscoder>()?;
    register_reference_sd(m)?;
    register_rate33(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
