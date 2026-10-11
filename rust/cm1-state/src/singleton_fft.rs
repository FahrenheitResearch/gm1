//! CM1 r21.1's `singleton.F`: R. C. Singleton's mixed-radix FFT as CM1
//! compiles it, in default REAL (`fftkind = KIND(0.0)`), statement by
//! statement.
//!
//! `poiss.F` transforms with `fft(rhs)` and `fft(rhs, inv=.true.)`, the 2-D
//! generic (`singleton.F:189-205`), which scales every element by
//! `sqrt(1/n)` (`singleton.F:357-361`) and runs `fftradix` along each
//! dimension, first (fastest) to last.  Every rounding step is CM1's:
//! Fortran's left-to-right evaluation, complex products as gfortran expands
//! them under its default `-fcx-fortran-rules` (`re = a*c - b*d`,
//! `im = a*d + b*c`, no FMA), `sinf`/`cosf` from the C library as gfortran
//! calls them.  `permute` moves words and computes none.
//!
//! Indices are CM1's one-based ones: every array here carries an unused
//! word 0 so a transcribed subscript is the source's subscript.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

/// A default-REAL complex word.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C32 {
    pub re: f32,
    pub im: f32,
}

impl C32 {
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    /// gfortran's inline complex product: (a*c - b*d, a*d + b*c).
    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    /// `CMPLX(-AIMAG(c), REAL(c))`.
    fn times_i(self) -> Self {
        Self::new(-self.im, self.re)
    }
    /// `CMPLX(AIMAG(c), -REAL(c))`.
    fn times_minus_i(self) -> Self {
        Self::new(self.im, -self.re)
    }
    /// `CMPLX(REAL(c)*s, AIMAG(c)*s)`.
    fn scale(self, s: f32) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}

// singleton.F:157-160, REAL(fftkind) parameters.
const SIN60: f32 = 0.866_025_403_784_438_65_f32;
const COS72: f32 = 0.309_016_994_374_947_42_f32;
const SIN72: f32 = 0.951_056_516_295_153_57_f32;
const PI: f32 = 3.141_592_653_589_793_23_f32;

/// `fft2d(array, inv=inv)` (`singleton.F:189-205` then `fftn`,
/// `singleton.F:327-373`) on a column-major `(n1, n2)` array: element
/// `(i, j)` (one-based) is `data[(j-1)*n1 + (i-1)]`.
pub fn fft2d(data: &mut [C32], n1: usize, n2: usize, inverse: bool) {
    assert_eq!(data.len(), n1 * n2, "fft2d extent");
    let ntotal = n1 * n2;
    // singleton.F:357: scale = SQRT(1.0_fftkind / PRODUCT(shape(d(1:ndim)))).
    let scale = (1.0_f32 / (ntotal as i32) as f32).sqrt();
    let mut array = vec![C32::default(); ntotal + 1];
    for (index, value) in data.iter().enumerate() {
        // singleton.F:358-361.
        array[index + 1] = C32::new(value.re * scale, value.im * scale);
    }
    // singleton.F:363-366: dimension 1 (nspan = n1), then 2 (nspan = n1*n2).
    fftradix(&mut array, ntotal as i64, n1 as i64, n1 as i64, inverse);
    fftradix(&mut array, ntotal as i64, n2 as i64, (n1 * n2) as i64, inverse);
    data.copy_from_slice(&array[1..]);
}

/// `singleton.F:376-397`.
fn fftradix(array: &mut [C32], ntotal: i64, npass: i64, nspan: i64, inv: bool) {
    if npass <= 1 {
        return;
    }
    let mut factor = [0_i64; 33];
    let (nfactor, nsquare) = factorize(npass, &mut factor);
    let maxfactor = (1..=nfactor).map(|i| factor[i as usize]).max().unwrap_or(1);
    transform(array, ntotal, npass, nspan, &factor, nfactor, inv);
    permute(array, ntotal, npass, nspan, &mut factor, nfactor, nsquare, maxfactor);
}

/// `singleton.F:403-463`: returns `(nfactor, nsquare)`; `factor` is one-based.
fn factorize(npass: i64, factor: &mut [i64; 33]) -> (i64, i64) {
    let mut nfactor = 0_i64;
    let mut nsquare;
    let mut k = npass;
    while k % 16 == 0 {
        nfactor += 1;
        factor[nfactor as usize] = 4;
        k /= 16;
    }
    let mut j = 3_i64;
    let mut jj = 9_i64;
    loop {
        while k % jj == 0 {
            nfactor += 1;
            factor[nfactor as usize] = j;
            k /= jj;
        }
        j += 2;
        jj = j * j;
        if jj > k {
            break;
        }
    }
    if k <= 4 {
        nsquare = nfactor;
        factor[(nfactor + 1) as usize] = k;
        if k != 1 {
            nfactor += 1;
        }
    } else {
        if k - ((k / 4) << 2) == 0 {
            nfactor += 1;
            factor[nfactor as usize] = 2;
            k /= 4;
        }
        nsquare = nfactor;
        j = 2;
        loop {
            if k % j == 0 {
                nfactor += 1;
                factor[nfactor as usize] = j;
                k /= j;
            }
            j = (((j + 1) / 2) << 1) + 1;
            if j > k {
                break;
            }
        }
    }
    if nsquare > 0 {
        j = nsquare;
        loop {
            nfactor += 1;
            factor[nfactor as usize] = factor[j as usize];
            j -= 1;
            if j == 0 {
                break;
            }
        }
    }
    let _ = &mut nsquare;
    (nfactor, nsquare)
}

/// `singleton.F:466-825`.
fn transform(array: &mut [C32], ntotal: i64, npass: i64, nspan: i64, factor: &[i64; 33], nfactor: i64, inv: bool) {
    let a = |array: &[C32], index: i64| array[index as usize];
    let maxfactor = (1..=nfactor).map(|i| factor[i as usize]).max().unwrap_or(1) as usize;
    let mut ctmp = vec![C32::default(); maxfactor + 1];
    let mut sine = vec![0.0_f32; maxfactor + 1];
    let mut cosine = vec![0.0_f32; maxfactor + 1];

    let c72 = COS72;
    let (s72, s60, mut pi2) = if inv { (SIN72, SIN60, PI) } else { (-SIN72, -SIN60, -PI) };

    let nt = ntotal;
    let nn = nt - 1;
    let mut kspan = nspan;
    let jc = nspan / npass;
    let radf = pi2 * jc as f32;
    pi2 *= 2.0_f32;

    let mut ii = 0_i64;
    let mut jf = 0_i64;
    let (mut c1, mut s1, mut c2, mut s2, mut c3, mut s3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    loop {
        let mut sd = radf / kspan as f32;
        let mut cd = sd.sin();
        cd = 2.0_f32 * cd * cd;
        sd = (sd + sd).sin();
        let mut kk = 1_i64;
        ii += 1;

        match factor[ii as usize] {
            2 => {
                // singleton.F:515-567.
                kspan /= 2;
                let mut k1 = kspan + 2;
                loop {
                    loop {
                        let k2 = kk + kspan;
                        let ck = a(array, k2);
                        array[k2 as usize] = a(array, kk).sub(ck);
                        array[kk as usize] = a(array, kk).add(ck);
                        kk = k2 + kspan;
                        if kk > nn {
                            break;
                        }
                    }
                    kk -= nn;
                    if kk > jc {
                        break;
                    }
                }
                if kk > kspan {
                    return;
                }
                loop {
                    c1 = 1.0_f32 - cd;
                    s1 = sd;
                    let mut k2;
                    loop {
                        loop {
                            loop {
                                k2 = kk + kspan;
                                let ck = a(array, kk).sub(a(array, k2));
                                array[kk as usize] = a(array, kk).add(a(array, k2));
                                array[k2 as usize] = ck.mul(C32::new(c1, s1));
                                kk = k2 + kspan;
                                if kk >= nt {
                                    break;
                                }
                            }
                            k2 = kk - nt;
                            c1 = -c1;
                            kk = k1 - k2;
                            if kk <= k2 {
                                break;
                            }
                        }
                        let ak = c1 - (cd * c1 + sd * s1);
                        s1 = sd * c1 - cd * s1 + s1;
                        c1 = 2.0_f32 - (ak * ak + s1 * s1);
                        s1 *= c1;
                        c1 *= ak;
                        kk += jc;
                        if kk >= k2 {
                            break;
                        }
                    }
                    k1 = k1 + 1 + 1;
                    kk = (k1 - kspan) / 2 + jc;
                    if kk > jc + jc {
                        break;
                    }
                }
            }
            4 => {
                // singleton.F:569-627.
                kspan /= 4;
                loop {
                    c1 = 1.0_f32;
                    s1 = 0.0_f32;
                    loop {
                        loop {
                            let k1 = kk + kspan;
                            let k2 = k1 + kspan;
                            let k3 = k2 + kspan;
                            let mut ckp = a(array, kk).add(a(array, k2));
                            let mut ckm = a(array, kk).sub(a(array, k2));
                            let mut cjp = a(array, k1).add(a(array, k3));
                            let cjm = a(array, k1).sub(a(array, k3));
                            array[kk as usize] = ckp.add(cjp);
                            cjp = ckp.sub(cjp);
                            if inv {
                                ckp = ckm.add(cjm.times_i());
                                ckm = ckm.add(cjm.times_minus_i());
                            } else {
                                ckp = ckm.add(cjm.times_minus_i());
                                ckm = ckm.add(cjm.times_i());
                            }
                            if s1 == 0.0_f32 {
                                array[k1 as usize] = ckp;
                                array[k2 as usize] = cjp;
                                array[k3 as usize] = ckm;
                            } else {
                                array[k1 as usize] = ckp.mul(C32::new(c1, s1));
                                array[k2 as usize] = cjp.mul(C32::new(c2, s2));
                                array[k3 as usize] = ckm.mul(C32::new(c3, s3));
                            }
                            kk = k3 + kspan;
                            if kk > nt {
                                break;
                            }
                        }
                        c2 = c1 - (cd * c1 + sd * s1);
                        s1 = sd * c1 - cd * s1 + s1;
                        c1 = 2.0_f32 - (c2 * c2 + s1 * s1);
                        s1 *= c1;
                        c1 *= c2;
                        c2 = c1 * c1 - s1 * s1;
                        s2 = 2.0_f32 * c1 * s1;
                        c3 = c2 * c1 - s2 * s1;
                        s3 = c2 * s1 + s2 * c1;
                        kk = kk - nt + jc;
                        if kk > kspan {
                            break;
                        }
                    }
                    kk = kk - kspan + 1;
                    if kk > jc {
                        break;
                    }
                }
                if kspan == jc {
                    return;
                }
            }
            _ => {
                // singleton.F:629-819: odd factors, then the rotation.
                let mut k = factor[ii as usize];
                let ispan = kspan;
                kspan /= k;
                match k {
                    3 => loop {
                        loop {
                            let k1 = kk + kspan;
                            let k2 = k1 + kspan;
                            let mut ck = a(array, kk);
                            let mut cj = a(array, k1).add(a(array, k2));
                            array[kk as usize] = ck.add(cj);
                            ck = ck.sub(C32::new(0.5_f32 * cj.re, 0.5_f32 * cj.im));
                            cj = C32::new(
                                (a(array, k1).re - a(array, k2).re) * s60,
                                (a(array, k1).im - a(array, k2).im) * s60,
                            );
                            array[k1 as usize] = ck.add(cj.times_i());
                            array[k2 as usize] = ck.add(cj.times_minus_i());
                            kk = k2 + kspan;
                            if kk >= nn {
                                break;
                            }
                        }
                        kk -= nn;
                        if kk > kspan {
                            break;
                        }
                    },
                    5 => {
                        c2 = c72 * c72 - s72 * s72;
                        s2 = 2.0_f32 * c72 * s72;
                        loop {
                            loop {
                                let k1 = kk + kspan;
                                let k2 = k1 + kspan;
                                let k3 = k2 + kspan;
                                let k4 = k3 + kspan;
                                let ckp = a(array, k1).add(a(array, k4));
                                let ckm = a(array, k1).sub(a(array, k4));
                                let cjp = a(array, k2).add(a(array, k3));
                                let cjm = a(array, k2).sub(a(array, k3));
                                let cc = a(array, kk);
                                array[kk as usize] = cc.add(ckp).add(cjp);
                                let mut ck = ckp.scale(c72).add(cjp.scale(c2)).add(cc);
                                let mut cj = ckm.scale(s72).add(cjm.scale(s2));
                                array[k1 as usize] = ck.add(cj.times_i());
                                array[k4 as usize] = ck.add(cj.times_minus_i());
                                ck = ckp.scale(c2).add(cjp.scale(c72)).add(cc);
                                cj = ckm.scale(s2).sub(cjm.scale(s72));
                                array[k2 as usize] = ck.add(cj.times_i());
                                array[k3 as usize] = ck.add(cj.times_minus_i());
                                kk = k4 + kspan;
                                if kk >= nn {
                                    break;
                                }
                            }
                            kk -= nn;
                            if kk > kspan {
                                break;
                            }
                        }
                    }
                    _ => {
                        if k != jf {
                            jf = k;
                            s1 = pi2 / k as f32;
                            c1 = s1.cos();
                            s1 = s1.sin();
                            cosine[jf as usize] = 1.0_f32;
                            sine[jf as usize] = 0.0_f32;
                            let mut j = 1_i64;
                            loop {
                                cosine[j as usize] = cosine[k as usize] * c1 + sine[k as usize] * s1;
                                sine[j as usize] = cosine[k as usize] * s1 - sine[k as usize] * c1;
                                k -= 1;
                                cosine[k as usize] = cosine[j as usize];
                                sine[k as usize] = -sine[j as usize];
                                j += 1;
                                if j >= k {
                                    break;
                                }
                            }
                        }
                        loop {
                            loop {
                                let mut k1 = kk;
                                let mut k2 = kk + ispan;
                                let cc = a(array, kk);
                                let mut ck = cc;
                                let mut j = 1_i64;
                                k1 += kspan;
                                loop {
                                    k2 -= kspan;
                                    j += 1;
                                    ctmp[j as usize] = a(array, k1).add(a(array, k2));
                                    ck = ck.add(ctmp[j as usize]);
                                    j += 1;
                                    ctmp[j as usize] = a(array, k1).sub(a(array, k2));
                                    k1 += kspan;
                                    if k1 >= k2 {
                                        break;
                                    }
                                }
                                array[kk as usize] = ck;
                                k1 = kk;
                                k2 = kk + ispan;
                                j = 1;
                                loop {
                                    k1 += kspan;
                                    k2 -= kspan;
                                    let mut jj = j;
                                    ck = cc;
                                    let mut cj = C32::new(0.0_f32, 0.0_f32);
                                    k = 1;
                                    loop {
                                        k += 1;
                                        ck = ck.add(ctmp[k as usize].scale(cosine[jj as usize]));
                                        k += 1;
                                        cj = cj.add(ctmp[k as usize].scale(sine[jj as usize]));
                                        jj += j;
                                        if jj > jf {
                                            jj -= jf;
                                        }
                                        if k >= jf {
                                            break;
                                        }
                                    }
                                    k = jf - j;
                                    array[k1 as usize] = ck.add(cj.times_i());
                                    array[k2 as usize] = ck.add(cj.times_minus_i());
                                    j += 1;
                                    if j >= k {
                                        break;
                                    }
                                }
                                kk += ispan;
                                if kk > nn {
                                    break;
                                }
                            }
                            kk -= nn;
                            if kk > kspan {
                                break;
                            }
                        }
                    }
                }
                // singleton.F:787-819: multiply by the rotation factor.
                if ii == nfactor {
                    return;
                }
                kk = jc + 1;
                loop {
                    c2 = 1.0_f32 - cd;
                    s1 = sd;
                    loop {
                        c1 = c2;
                        s2 = s1;
                        kk += kspan;
                        loop {
                            loop {
                                array[kk as usize] = C32::new(c2, s2).mul(a(array, kk));
                                kk += ispan;
                                if kk > nt {
                                    break;
                                }
                            }
                            let ak = s1 * s2;
                            s2 = s1 * c2 + c1 * s2;
                            c2 = c1 * c2 - ak;
                            kk = kk - nt + kspan;
                            if kk > ispan {
                                break;
                            }
                        }
                        c2 = c1 - (cd * c1 + sd * s1);
                        s1 = s1 + sd * c1 - cd * s1;
                        c1 = 2.0_f32 - (c2 * c2 + s1 * s1);
                        s1 *= c1;
                        c2 *= c1;
                        kk = kk - ispan + jc;
                        if kk > kspan {
                            break;
                        }
                    }
                    kk = kk - kspan + jc + 1;
                    if kk > jc + jc {
                        break;
                    }
                }
            }
        }
    }
}

/// `singleton.F:828-1068`: the words back into natural order.
#[allow(clippy::too_many_arguments)]
fn permute(
    array: &mut [C32],
    ntotal: i64,
    npass: i64,
    nspan: i64,
    factor: &mut [i64; 33],
    nfactor: i64,
    nsquare: i64,
    maxfactor: i64,
) {
    let mut ctmp = vec![C32::default(); maxfactor as usize + 1];
    let nperm = if nfactor - (nsquare << 1) > 0 {
        let product: i64 = ((nsquare + 1)..=(nfactor - nsquare)).map(|i| factor[i as usize]).product();
        (nfactor + 1).max(product - 1)
    } else {
        nfactor + 1
    };
    let mut perm = vec![0_i64; nperm as usize + 2];

    let mut nt = ntotal;
    let mut kt = nsquare;
    let mut kspan = nspan;
    let mut jc = nspan / npass;

    perm[1] = nspan;
    // k3 keeps the square-factor stage's value into the cycle stage, as in
    // the source (singleton.F:957 and :1004 are its only assignments).
    let (mut j, mut k, mut kk, mut k2);
    let mut k3 = 0_i64;
    if kt > 0 {
        k = kt + kt + 1;
        if nfactor < k {
            k -= 1;
        }
        j = 1;
        perm[(k + 1) as usize] = jc;
        loop {
            perm[(j + 1) as usize] = perm[j as usize] / factor[j as usize];
            perm[k as usize] = perm[(k + 1) as usize] * factor[j as usize];
            j += 1;
            k -= 1;
            if j >= k {
                break;
            }
        }
        k3 = perm[(k + 1) as usize];
        kspan = perm[2];
        kk = jc + 1;
        k2 = kspan + 1;
        j = 1;

        if npass != ntotal {
            'permute_multi: loop {
                loop {
                    loop {
                        k = kk + jc;
                        loop {
                            array.swap(kk as usize, k2 as usize);
                            kk += 1;
                            k2 += 1;
                            if kk >= k {
                                break;
                            }
                        }
                        kk = kk + nspan - jc;
                        k2 = k2 + nspan - jc;
                        if kk >= nt {
                            break;
                        }
                    }
                    kk = kk - nt + jc;
                    k2 = k2 - nt + kspan;
                    if k2 >= nspan {
                        break;
                    }
                }
                loop {
                    loop {
                        k2 -= perm[j as usize];
                        j += 1;
                        k2 += perm[(j + 1) as usize];
                        if k2 <= perm[j as usize] {
                            break;
                        }
                    }
                    j = 1;
                    loop {
                        if kk < k2 {
                            continue 'permute_multi;
                        }
                        kk += jc;
                        k2 += kspan;
                        if k2 >= nspan {
                            break;
                        }
                    }
                    if kk >= nspan {
                        break;
                    }
                }
                break;
            }
        } else {
            'permute_single: loop {
                loop {
                    array.swap(kk as usize, k2 as usize);
                    kk += 1;
                    k2 += kspan;
                    if k2 >= nspan {
                        break;
                    }
                }
                loop {
                    loop {
                        k2 -= perm[j as usize];
                        j += 1;
                        k2 += perm[(j + 1) as usize];
                        if k2 <= perm[j as usize] {
                            break;
                        }
                    }
                    j = 1;
                    loop {
                        if kk < k2 {
                            continue 'permute_single;
                        }
                        kk += 1;
                        k2 += kspan;
                        if k2 >= nspan {
                            break;
                        }
                    }
                    if kk >= nspan {
                        break;
                    }
                }
                break;
            }
        }
        jc = k3;
    }

    if (kt << 1) + 1 >= nfactor {
        return;
    }

    let ispan = perm[(kt + 1) as usize];
    // singleton.F:966-973: permutation for square-free factors of n.
    j = nfactor - kt;
    factor[(j + 1) as usize] = 1;
    loop {
        factor[j as usize] *= factor[(j + 1) as usize];
        j -= 1;
        if j == kt {
            break;
        }
    }
    kt += 1;
    let nn = factor[kt as usize] - 1;
    j = 0;
    let mut jj = 0_i64;
    loop {
        k = kt + 1;
        k2 = factor[kt as usize];
        kk = factor[k as usize];
        j += 1;
        if j > nn {
            break;
        }
        jj += kk;
        while jj >= k2 {
            jj -= k2;
            k2 = kk;
            k += 1;
            kk = factor[k as usize];
            jj += kk;
        }
        perm[j as usize] = jj;
    }
    // singleton.F:991-1009: the permutation cycles of length greater than 1.
    j = 0;
    loop {
        loop {
            j += 1;
            kk = perm[j as usize];
            if kk >= 0 {
                break;
            }
        }
        if kk != j {
            loop {
                k = kk;
                kk = perm[k as usize];
                perm[k as usize] = -kk;
                if kk == j {
                    break;
                }
            }
            k3 = kk;
        } else {
            perm[j as usize] = -j;
            if j == nn {
                break;
            }
        }
    }
    // singleton.F:1010-1065: reorder, following the cycles.
    loop {
        j = k3 + 1;
        nt -= ispan;
        let ii = nt - 1 + 1;
        if nt < 0 {
            break;
        }
        loop {
            loop {
                j -= 1;
                if perm[j as usize] >= 0 {
                    break;
                }
            }
            jj = jc;
            loop {
                kspan = jj;
                if jj > maxfactor {
                    kspan = maxfactor;
                }
                jj -= kspan;
                k = perm[j as usize];
                kk = jc * k + ii + jj;
                let mut k1 = kk + kspan;
                k2 = 0;
                loop {
                    k2 += 1;
                    ctmp[k2 as usize] = array[k1 as usize];
                    k1 -= 1;
                    if k1 == kk {
                        break;
                    }
                }
                loop {
                    k1 = kk + kspan;
                    k2 = k1 - jc * (k + perm[k as usize]);
                    k = -perm[k as usize];
                    loop {
                        array[k1 as usize] = array[k2 as usize];
                        k1 -= 1;
                        k2 -= 1;
                        if k1 == kk {
                            break;
                        }
                    }
                    kk = k2;
                    if k == j {
                        break;
                    }
                }
                k1 = kk + kspan;
                k2 = 0;
                loop {
                    k2 += 1;
                    array[k1 as usize] = ctmp[k2 as usize];
                    k1 -= 1;
                    if k1 == kk {
                        break;
                    }
                }
                if jj == 0 {
                    break;
                }
            }
            if j == 1 {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft2(data: &[C32], n1: usize, n2: usize, inverse: bool) -> Vec<(f64, f64)> {
        let sign = if inverse { 1.0 } else { -1.0 };
        let scale = 1.0 / ((n1 * n2) as f64).sqrt();
        let mut out = vec![(0.0, 0.0); n1 * n2];
        for q in 0..n2 {
            for p in 0..n1 {
                let (mut re, mut im) = (0.0, 0.0);
                for j in 0..n2 {
                    for i in 0..n1 {
                        let v = data[j * n1 + i];
                        let angle = sign * 2.0 * std::f64::consts::PI
                            * ((p * i) as f64 / n1 as f64 + (q * j) as f64 / n2 as f64);
                        let (s, c) = angle.sin_cos();
                        re += f64::from(v.re) * c - f64::from(v.im) * s;
                        im += f64::from(v.re) * s + f64::from(v.im) * c;
                    }
                }
                out[q * n1 + p] = (re * scale, im * scale);
            }
        }
        out
    }

    #[test]
    fn every_factor_path_is_the_discrete_fourier_transform() {
        // 4s, 2, 3, 5, 7 (the general odd arm), square and square-free
        // factors, and CM1's mirrored and periodic extents.
        for (n1, n2) in [(64, 64), (32, 32), (60, 12), (6, 10), (7, 9), (12, 5), (1, 8), (49, 3), (22, 14), (30, 45)] {
            let data: Vec<C32> = (0..n1 * n2)
                .map(|n| C32::new(((n as f32) * 0.37).sin(), ((n as f32) * 0.11).cos()))
                .collect();
            for inverse in [false, true] {
                let mut ours = data.clone();
                fft2d(&mut ours, n1, n2, inverse);
                let exact = dft2(&data, n1, n2, inverse);
                for (m, (a, b)) in ours.iter().zip(&exact).enumerate() {
                    let tolerance = 2e-5 * (1.0 + b.0.abs() + b.1.abs());
                    assert!(
                        (f64::from(a.re) - b.0).abs() < tolerance && (f64::from(a.im) - b.1).abs() < tolerance,
                        "({n1},{n2}) inverse={inverse} word {m}: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }
}
