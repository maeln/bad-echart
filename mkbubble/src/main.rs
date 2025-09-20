use std::{
    collections::{BTreeSet, HashSet},
    fmt::Display,
    path::PathBuf,
};

use bpaf::{Parser, positional};
use image::{DynamicImage, GenericImageView};
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::{Rng, SeedableRng};
use rayon::iter::{ParallelBridge, ParallelIterator};

const MAX_RADIUS: u32 = 25;

#[derive(Debug, Clone)]
struct Circle {
    x: u32,
    y: u32,
    r: u32,
}

impl Display for Circle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("[{},{},{}]", self.x, self.y, self.r))
    }
}

fn u32_to_ratio(value: &u32) -> f32 {
    0.7 + (value - 4) as f32 * (0.98 - 0.6) / (MAX_RADIUS - 4) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = positional::<PathBuf>("IMG")
        .help("frame to analyze")
        .to_options()
        .run();

    let img = image::open(&path)?;

    let targets_pixels: Vec<(u32, u32)> = (0..(img.width() * img.height()))
        .map(|d1| (d1 % img.width(), d1 / img.width()))
        .par_bridge()
        .filter(|coord| {
            let pixel = img.get_pixel(coord.0, coord.1).0;
            let luminance = calculate_luminance(pixel[0], pixel[1], pixel[2]);
            return luminance > 0.5;
        })
        .collect();
    // targets_pixels.sort();

    // let mut pixel_tree = BTreeSet::new();
    // for px in targets_pixels {
    //     pixel_tree.insert(px);
    // }

    // let mut rng = StdRng::seed_from_u64(1337);
    // let mut rng = rand::rng();
    // let choosen = targets_pixels.into_iter().choose_multiple(&mut rng, TAPS);
    let mut choosen_set = HashSet::new();
    for p in targets_pixels {
        choosen_set.insert(p);
    }

    let mut circles: Vec<Circle> = Vec::new();
    // let mut i = 0;
    while choosen_set.len() > 0 {
        // let coord = if i % 2 == 0 {
        //     pixel_tree.pop_last().unwrap()
        // } else {
        //     pixel_tree.pop_first().unwrap()
        // };
        let coord = choosen_set.iter().next().unwrap().clone();
        choosen_set.remove(&coord);
        // i += 1;
        let (circle, to_rm) = find_biggest_circle(&img, coord.0, coord.1);
        if circle.r < 2 {
            // ignore small circle.
            continue;
        }
        // let ratio = u32_to_ratio(&circle.r);
        circles.push(circle);
        // We want to allow for some overlap so we let some circle withing through
        // let maxdl = (to_rm.len() as f32 * ratio).ceil() as usize;
        // let exclude = to_rm.into_iter().choose_multiple(&mut rng, maxdl);
        for px in to_rm {
            choosen_set.remove(&px);
        }
    }

    let circle_fmt = circles
        .iter()
        .map(|c| Circle {
            y: (img.height() - c.y),
            x: c.x,
            r: c.r,
        }) // In echart, y=0 is at the bottom
        .map(|c| c.to_string())
        .collect::<Vec<String>>()
        .join(",");
    let arr = format!("[{}],", circle_fmt);
    println!("{}", arr);

    Ok(())
}

fn calculate_luminance(r: u8, g: u8, b: u8) -> f32 {
    // Using the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    let r_norm = r as f32 / 255.0;
    let g_norm = g as f32 / 255.0;
    let b_norm = b as f32 / 255.0;
    0.299 * r_norm + 0.587 * g_norm + 0.114 * b_norm
}

fn find_biggest_circle(img: &DynamicImage, cx: u32, cy: u32) -> (Circle, Vec<(u32, u32)>) {
    let mut circle = Circle { x: cx, y: cy, r: 1 };
    for r in 3..=MAX_RADIUS {
        let valid = check_circle_brightness(img, cx, cy, r);
        if !valid {
            break;
        }
        circle.r = r;
    }
    let fr = circle.r.clone();
    if circle.r > 3 {
        // Allow for a bit of overlap
        return (circle, pixels_in_circle(cx, cy, fr - 1));
    }
    (circle, vec![])
}

fn pixels_in_circle(cx: u32, cy: u32, r: u32) -> Vec<(u32, u32)> {
    let mut pixels = Vec::new();
    let cx = cx as i32;
    let cy = cy as i32;
    let radius = r as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 {
                    // We ignore out of bound pixels
                    continue;
                }
                pixels.push((x as u32, y as u32));
            }
        }
    }
    pixels
}

fn check_circle_brightness(img: &DynamicImage, cx: u32, cy: u32, radius: u32) -> bool {
    let cx = cx as i32;
    let cy = cy as i32;
    let radius = radius as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 || x >= img.width() as i32 || y >= img.height() as i32 {
                    // We ignore out of bound pixels
                    continue;
                }
                let pixel = img.get_pixel(x as u32, y as u32);
                let luminance = calculate_luminance(pixel[0], pixel[1], pixel[2]);

                if luminance <= 0.5 {
                    return false;
                }
            }
        }
    }

    true
}
