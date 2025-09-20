use std::{collections::HashSet, fmt::Display, path::PathBuf};

use bpaf::{Parser, positional};
use image::{DynamicImage, GenericImageView};
use rand::seq::IteratorRandom;
use rayon::iter::{ParallelBridge, ParallelIterator};

const TAPS: usize = 50_000;
const MAX_RADIUS: u32 = 20;

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

    let mut rng = rand::rng();
    let choosen = targets_pixels.into_iter().choose_multiple(&mut rng, TAPS);
    let mut choosen_set = HashSet::new();
    for p in choosen {
        choosen_set.insert(p);
    }

    let mut circles: Vec<Circle> = Vec::new();
    while choosen_set.len() > 0 {
        let coord = choosen_set.iter().next().unwrap().clone();
        let (circle, to_rm) = find_biggest_circle(&img, coord.0, coord.1);
        if circle.r < 3 {
            // ignore small circle.
            choosen_set.remove(&coord);
            continue;
        }
        circles.push(circle);
        // We want to allow for some overlap so we let some circle withing through
        let maxdl = (to_rm.len() as f32 * 0.8).ceil() as usize;
        let exclude = to_rm.into_iter().choose_multiple(&mut rng, maxdl);
        for px in exclude {
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
    let mut pixels_rm: Vec<(u32, u32)> = Vec::new();
    for r in 1..=MAX_RADIUS {
        let (valid, pix) = check_circle_brightness(img, cx, cy, r);
        if !valid {
            break;
        }
        circle.r = r;
        pixels_rm = pix;
    }
    (circle, pixels_rm)
}

fn check_circle_brightness(
    img: &DynamicImage,
    cx: u32,
    cy: u32,
    radius: u32,
) -> (bool, Vec<(u32, u32)>) {
    let mut included: HashSet<(u32, u32)> = HashSet::new();
    included.insert((cx, cy));
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
                    return (false, vec![]);
                } else if x >= 0 && y >= 0 {
                    included.insert((x as u32, y as u32));
                }
            }
        }
    }

    (true, included.into_iter().collect())
}
