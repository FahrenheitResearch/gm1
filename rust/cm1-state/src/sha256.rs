//! Small, dependency-free SHA-256 implementation for container receipts.
//!
//! Multi-gibibyte checkpoint payloads are hashed several times per
//! publication, so on x86-64 processors with the SHA extensions the block
//! compression runs through the `sha256rnds2`/`sha256msg1`/`sha256msg2`
//! instructions (detected at run time).  Both paths compute FIPS 180-4
//! SHA-256; the tests compare them on every length that crosses a block edge.

const INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

#[derive(Clone)]
pub(crate) struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    message_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self {
            state: INITIAL_STATE,
            buffer: [0; 64],
            buffer_len: 0,
            message_len: 0,
        }
    }
}

impl Sha256 {
    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.message_len = self
            .message_len
            .checked_add(bytes.len() as u64)
            .expect("SHA-256 input length overflow");

        if self.buffer_len != 0 {
            let needed = 64 - self.buffer_len;
            let copied = needed.min(bytes.len());
            self.buffer[self.buffer_len..self.buffer_len + copied]
                .copy_from_slice(&bytes[..copied]);
            self.buffer_len += copied;
            bytes = &bytes[copied..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffer_len = 0;
            } else {
                // The complete input was appended to the existing partial
                // block.  Preserve that block for the next update instead of
                // falling through and resetting `buffer_len` to zero.
                return;
            }
        }

        let whole = bytes.len() - bytes.len() % 64;
        if whole != 0 {
            compress_blocks(&mut self.state, &bytes[..whole]);
            bytes = &bytes[whole..];
        }

        self.buffer[..bytes.len()].copy_from_slice(bytes);
        self.buffer_len = bytes.len();
    }

    pub(crate) fn finalize(mut self) -> [u8; 32] {
        let bit_len = self
            .message_len
            .checked_mul(8)
            .expect("SHA-256 bit length overflow");
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            self.buffer[self.buffer_len..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
        } else {
            self.buffer[self.buffer_len..56].fill(0);
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);

        let mut digest = [0; 32];
        for (chunk, word) in digest.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn compress(&mut self, block: &[u8; 64]) {
        compress_blocks(&mut self.state, block);
    }
}

/// Compress every complete 64-byte block of `blocks` into `state`.
fn compress_blocks(state: &mut [u32; 8], blocks: &[u8]) {
    debug_assert_eq!(blocks.len() % 64, 0);
    #[cfg(target_arch = "x86_64")]
    {
        if shani::available() {
            // SAFETY: the SHA, SSSE3 and SSE4.1 features the function is
            // compiled for were detected on this processor just above.
            unsafe { shani::compress_blocks(state, blocks) };
            return;
        }
    }
    compress_blocks_portable(state, blocks);
}

fn compress_blocks_portable(state: &mut [u32; 8], blocks: &[u8]) {
    for block in blocks.chunks_exact(64) {
        compress_block_portable(state, block.try_into().expect("fixed block length"));
    }
}

fn compress_block_portable(state: &mut [u32; 8], block: &[u8; 64]) {
    {
        let mut words = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes(chunk.try_into().expect("fixed word length"));
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for index in 0..64 {
            let upper_e = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let first = h
                .wrapping_add(upper_e)
                .wrapping_add(choose)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(words[index]);
            let upper_a = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = upper_a.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(first);
            d = c;
            c = b;
            b = a;
            a = first.wrapping_add(second);
        }

        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod shani {
    use super::ROUND_CONSTANTS;
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    static DETECTED: AtomicU8 = AtomicU8::new(0);

    pub(super) fn available() -> bool {
        match DETECTED.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let found = std::arch::is_x86_feature_detected!("sha")
                    && std::arch::is_x86_feature_detected!("ssse3")
                    && std::arch::is_x86_feature_detected!("sse4.1");
                DETECTED.store(if found { 1 } else { 2 }, Ordering::Relaxed);
                found
            }
        }
    }

    #[inline(always)]
    unsafe fn constants(round: usize) -> __m128i {
        _mm_set_epi32(
            ROUND_CONSTANTS[round + 3] as i32,
            ROUND_CONSTANTS[round + 2] as i32,
            ROUND_CONSTANTS[round + 1] as i32,
            ROUND_CONSTANTS[round] as i32,
        )
    }

    /// Four rounds on (ABEF, CDGH) with message words `w` (element 0 first).
    #[inline(always)]
    unsafe fn rounds4(abef: &mut __m128i, cdgh: &mut __m128i, w: __m128i, round: usize) {
        let words = _mm_add_epi32(w, constants(round));
        *cdgh = _mm_sha256rnds2_epu32(*cdgh, *abef, words);
        *abef = _mm_sha256rnds2_epu32(*abef, *cdgh, _mm_shuffle_epi32(words, 0x0e));
    }

    /// W[t+16..t+20] from W[t..t+4], W[t+4..t+8], W[t+8..t+12], W[t+12..t+16].
    #[inline(always)]
    unsafe fn schedule(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> __m128i {
        let sum = _mm_add_epi32(_mm_sha256msg1_epu32(v0, v1), _mm_alignr_epi8(v3, v2, 4));
        _mm_sha256msg2_epu32(sum, v3)
    }

    #[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
    pub(super) unsafe fn compress_blocks(state: &mut [u32; 8], blocks: &[u8]) {
        // Byte-swap each 32-bit word: the message is big-endian.
        let mask = _mm_set_epi64x(0x0c0d_0e0f_0809_0a0b, 0x0405_0607_0001_0203);
        // Element order (0..3): ABEF = [F, E, B, A], CDGH = [H, G, D, C].
        let mut abef = _mm_set_epi32(
            state[0] as i32,
            state[1] as i32,
            state[4] as i32,
            state[5] as i32,
        );
        let mut cdgh = _mm_set_epi32(
            state[2] as i32,
            state[3] as i32,
            state[6] as i32,
            state[7] as i32,
        );
        for block in blocks.chunks_exact(64) {
            let saved_abef = abef;
            let saved_cdgh = cdgh;
            let pointer = block.as_ptr();
            // Message words W[0..4], W[4..8], W[8..12], W[12..16].
            let mut w = [
                _mm_shuffle_epi8(_mm_loadu_si128(pointer.cast()), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(pointer.add(16).cast()), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(pointer.add(32).cast()), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(pointer.add(48).cast()), mask),
            ];
            for (index, words) in w.iter().enumerate() {
                rounds4(&mut abef, &mut cdgh, *words, 4 * index);
            }
            // Rounds 16..64: each schedule step derives the next four words
            // from the previous sixteen, held as a ring of four vectors.
            let mut oldest = 0usize;
            for group in 4..16 {
                let next = schedule(
                    w[oldest],
                    w[(oldest + 1) % 4],
                    w[(oldest + 2) % 4],
                    w[(oldest + 3) % 4],
                );
                w[oldest] = next;
                oldest = (oldest + 1) % 4;
                rounds4(&mut abef, &mut cdgh, next, 4 * group);
            }
            abef = _mm_add_epi32(abef, saved_abef);
            cdgh = _mm_add_epi32(cdgh, saved_cdgh);
        }
        state[0] = _mm_extract_epi32(abef, 3) as u32;
        state[1] = _mm_extract_epi32(abef, 2) as u32;
        state[4] = _mm_extract_epi32(abef, 1) as u32;
        state[5] = _mm_extract_epi32(abef, 0) as u32;
        state[2] = _mm_extract_epi32(cdgh, 3) as u32;
        state[3] = _mm_extract_epi32(cdgh, 2) as u32;
        state[6] = _mm_extract_epi32(cdgh, 1) as u32;
        state[7] = _mm_extract_epi32(cdgh, 0) as u32;
    }
}

pub(crate) fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::default();
    hasher.update(bytes);
    hasher.finalize()
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

pub(crate) fn digest_hex(bytes: &[u8]) -> String {
    hex(&digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::{digest_hex, hex, Sha256};

    #[test]
    fn standard_vectors() {
        assert_eq!(
            digest_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest_hex(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn the_dispatched_compression_equals_the_portable_one() {
        // Every length across several block edges, with non-repeating bytes,
        // through whichever path this processor dispatches to, against the
        // portable FIPS 180-4 compression.
        let bytes = (0..1_031u32)
            .map(|value| (value.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect::<Vec<_>>();
        for length in 0..bytes.len() {
            let whole = length - length % 64;
            let mut dispatched = super::INITIAL_STATE;
            let mut portable = super::INITIAL_STATE;
            super::compress_blocks(&mut dispatched, &bytes[..whole]);
            super::compress_blocks_portable(&mut portable, &bytes[..whole]);
            assert_eq!(dispatched, portable, "length={length}");
        }
        let mut hasher = Sha256::default();
        hasher.update(&bytes);
        let mut state = super::INITIAL_STATE;
        let mut padded = bytes.clone();
        let bit_length = (bytes.len() as u64) * 8;
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&bit_length.to_be_bytes());
        super::compress_blocks_portable(&mut state, &padded);
        let mut expected = [0u8; 32];
        for (chunk, word) in expected.chunks_exact_mut(4).zip(state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        assert_eq!(hasher.finalize(), expected);
    }

    #[test]
    fn split_updates_preserve_partial_blocks() {
        let bytes = (0..173u16).map(|value| value as u8).collect::<Vec<_>>();
        let expected = digest_hex(&bytes);
        for split in 0..=bytes.len() {
            let mut hasher = Sha256::default();
            hasher.update(&bytes[..split]);
            hasher.update(&bytes[split..]);
            assert_eq!(hex(&hasher.finalize()), expected, "split={split}");
        }

        let mut hasher = Sha256::default();
        for chunk in bytes.chunks(7) {
            hasher.update(chunk);
        }
        assert_eq!(hex(&hasher.finalize()), expected);
    }
}
