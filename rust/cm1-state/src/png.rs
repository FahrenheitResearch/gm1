//! Minimal dependency-free RGB PNG encoder and bitmap drawing surface.

pub(crate) struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Canvas {
    pub(crate) fn new(width: usize, height: usize, color: [u8; 3]) -> Self {
        let mut pixels = vec![0_u8; width * height * 3];
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.copy_from_slice(&color);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn set(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let offset = (y as usize * self.width + x as usize) * 3;
        self.pixels[offset..offset + 3].copy_from_slice(&color);
    }

    pub(crate) fn fill_rect(&mut self, x: i32, y: i32, width: i32, height: i32, color: [u8; 3]) {
        for py in y..y + height {
            for px in x..x + width {
                self.set(px, py, color);
            }
        }
    }

    pub(crate) fn rect(&mut self, x: i32, y: i32, width: i32, height: i32, color: [u8; 3]) {
        self.line(x, y, x + width - 1, y, color);
        self.line(x, y + height - 1, x + width - 1, y + height - 1, color);
        self.line(x, y, x, y + height - 1, color);
        self.line(x + width - 1, y, x + width - 1, y + height - 1, color);
    }

    pub(crate) fn line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.set(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x0 += sx;
            }
            if twice <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    pub(crate) fn text(&mut self, x: i32, y: i32, text: &str, scale: i32, color: [u8; 3]) {
        let mut cursor = x;
        for character in text.chars() {
            let glyph = glyph(character.to_ascii_uppercase());
            for (row, bits) in glyph.iter().enumerate() {
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        self.fill_rect(
                            cursor + column * scale,
                            y + row as i32 * scale,
                            scale,
                            scale,
                            color,
                        );
                    }
                }
            }
            cursor += 6 * scale;
        }
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut scanlines = Vec::with_capacity((self.width * 3 + 1) * self.height);
        for row in self.pixels.chunks_exact(self.width * 3) {
            scanlines.push(0);
            scanlines.extend_from_slice(row);
        }
        let compressed = zlib_store(&scanlines);
        let mut png = Vec::with_capacity(compressed.len() + 128);
        png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&(self.width as u32).to_be_bytes());
        ihdr.extend_from_slice(&(self.height as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &compressed);
        chunk(&mut png, b"IEND", &[]);
        png
    }
}

fn chunk(output: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    output.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn zlib_store(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() + input.len() / 65_535 * 5 + 8);
    output.extend_from_slice(&[0x78, 0x01]);
    let mut offset = 0;
    while offset < input.len() {
        let length = (input.len() - offset).min(65_535);
        let final_block = offset + length == input.len();
        output.push(if final_block { 0x01 } else { 0x00 });
        output.extend_from_slice(&(length as u16).to_le_bytes());
        output.extend_from_slice(&(!(length as u16)).to_le_bytes());
        output.extend_from_slice(&input[offset..offset + length]);
        offset += length;
    }
    output.extend_from_slice(&adler32(input).to_be_bytes());
    output
}

fn adler32(bytes: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;
    let mut a = 1_u32;
    let mut b = 0_u32;
    for byte in bytes {
        a = (a + u32::from(*byte)) % MODULUS;
        b = (b + a) % MODULUS;
    }
    (b << 16) | a
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320_u32 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn glyph(character: char) -> [u8; 7] {
    match character {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 2, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        ':' => [0, 4, 4, 0, 4, 4, 0],
        '.' => [0, 0, 0, 0, 0, 4, 4],
        ',' => [0, 0, 0, 0, 4, 4, 8],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        '=' => [0, 31, 0, 31, 0, 0, 0],
        '(' => [2, 4, 8, 8, 8, 4, 2],
        ')' => [8, 4, 2, 2, 2, 4, 8],
        '+' => [0, 4, 4, 31, 4, 4, 0],
        ' ' => [0; 7],
        _ => [31, 1, 2, 4, 0, 4, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_a_valid_png_envelope() {
        let mut canvas = Canvas::new(8, 8, [255, 255, 255]);
        canvas.line(0, 0, 7, 7, [0, 0, 0]);
        let png = canvas.encode();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    }
}
