use libafl_bolts::rands::Rand;

pub(crate) trait RandExt: Rand {
    fn probability(&mut self, p: f64) -> bool {
        assert!((0.0..=1.0).contains(&p));
        p * 10000f64 >= self.below(10001) as f64
    }

    fn random_i32(&mut self) -> i32 {
        let interesting = &[
            -2147483648,
            -2147483647, // Int32 min
            -1073741824,
            -536870912,
            -268435456, // -2**32 / {4, 8, 16}
            -65537,
            -65536,
            -65535, // -2**16
            -4096,
            -1024,
            -256,
            -128, // Other powers of two
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            16,
            64, // Numbers around 0
            127,
            128,
            129, // 2**7
            255,
            256,
            257, // 2**8
            512,
            1000,
            1024,
            4096,
            10000, // Misc numbers
            65535,
            65536,
            65537, // 2**16
            268435456,
            536870912,
            1073741824, // 2**32 / {4, 8, 16}
            1073741823,
            1073741824,
            1073741825, // 2**30
            2147483647, // Int32 max
        ];

        if self.probability(0.5) {
            *self.choose(interesting)
        } else {
            self.next() as i32
        }
    }

    fn random_u32(&mut self) -> u32 {
        let interesting = &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 64, // Numbers around 0
            127, 128, 129, // 2**7
            255, 256, 257, // 2**8
            512, 1000, 1024, 4096, 10000, // Misc numbers
            65535, 65536, 65537, // 2**16
            268435456, 536870912, 1073741824, // 2**32 / {4, 8, 16}
            1073741823, 1073741824, 1073741825, // 2**30
            2147483647, 2147483648, 2147483649, // Int32 max
            4294967295, // Uint32 max
        ];

        if self.probability(0.5) {
            *self.choose(interesting)
        } else {
            self.between(0, u32::MAX as u64) as u32
        }
    }

    fn random_f32(&mut self) -> f32 {
        let interesting = &[
            -f32::MAX,
            -1e-15,
            -1e12,
            -1e9,
            -1e6,
            -1e3,
            -5.0,
            -4.0,
            -3.0,
            -2.0,
            -1.0,
            -f32::EPSILON,
            -f32::MIN,
            -0.0,
            0.0,
            f32::MIN,
            f32::EPSILON,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            1e3,
            1e6,
            1e9,
            1e12,
            1e-15,
            f32::MAX,
        ];

        if self.probability(0.5) {
            *self.choose(interesting)
        } else {
            self.next() as f32
        }
    }

    fn random_bool(&mut self) -> bool {
        self.between(0, 1) == 0
    }

    fn random_u8(&mut self) -> u8 {
        self.next() as u8
    }

    fn shuffle<T>(&mut self, s: &mut [T]) {
        let n = s.len() as u64;
        if n < 2 {
            return;
        }
        for i in 0..=n - 2 {
            let j = self.between(i, n - 1);
            s.swap(i as usize, j as usize);
        }
    }
}

impl<R: Rand> RandExt for R {}
