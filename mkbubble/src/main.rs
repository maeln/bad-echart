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
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

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

fn calculate_luminance(r: u8, g: u8, b: u8) -> f32 {
    // Using the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    let r_norm = r as f32 / 255.0;
    let g_norm = g as f32 / 255.0;
    let b_norm = b as f32 / 255.0;
    0.299 * r_norm + 0.587 * g_norm + 0.114 * b_norm
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = positional::<PathBuf>("IMG")
        .help("frame to analyze")
        .to_options()
        .run();

    let img = image::open(&path)?;

    let targets_pixels: HashSet<(u32, u32)> = (0..(img.width() * img.height()))
        .map(|d1| (d1 % img.width(), d1 / img.width()))
        .par_bridge()
        .filter(|coord| {
            let pixel = img.get_pixel(coord.0, coord.1).0;
            let luminance = calculate_luminance(pixel[0], pixel[1], pixel[2]);
            return luminance > 0.5;
        })
        .collect();

    let mut circles: Vec<Circle> = Vec::new();
    let mut first_pass_pixels = targets_pixels.clone();
    let mut second_pass_pixels = targets_pixels.clone();
    let mut second_pass_mask = first_pass_pixels.clone();

    // First pass: We only care about fitting as many big circle as possible
    while first_pass_pixels.len() > 0 {
        let coord = first_pass_pixels.iter().next().unwrap().clone();
        first_pass_pixels.remove(&coord);
        let circle = find_biggest_circle(&targets_pixels, coord.0, coord.1);
        if circle.r < MAX_RADIUS {
            continue;
        }
        let to_rm = pixels_in_circle(circle.x, circle.y, circle.r);
        circles.push(circle);
        for px in to_rm {
            first_pass_pixels.remove(&px);
        }
    }

    // Second pass: We make a new image from the first with the circles
    // of the first pass removed, but we make them a bit smaller to allow for some overlap
    // and we again find circles
    for circle in circles.iter() {
        pixels_in_circle(circle.x, circle.y, circle.r - 2)
            .iter()
            .for_each(|p| {
                second_pass_mask.remove(p);
            });
    }
    while second_pass_pixels.len() > 0 {
        let coord = second_pass_pixels.iter().next().unwrap().clone();
        second_pass_pixels.remove(&coord);
        let circle = find_biggest_circle(&second_pass_mask, coord.0, coord.1);
        if circle.r < 2 {
            continue;
        }
        let to_rm = pixels_in_circle(circle.x, circle.y, circle.r);
        circles.push(circle);
        for px in to_rm {
            second_pass_mask.remove(&px);
        }
    }

    // Output the circles.
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

fn find_biggest_circle(valid_px: &HashSet<(u32, u32)>, cx: u32, cy: u32) -> Circle {
    let mut circle = Circle { x: cx, y: cy, r: 1 };
    for r in 3..=MAX_RADIUS {
        let valid = check_circle(valid_px, cx, cy, r);
        if !valid {
            break;
        }
        circle.r = r;
    }
    circle
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

fn check_circle(valid_px: &HashSet<(u32, u32)>, cx: u32, cy: u32, radius: u32) -> bool {
    let cx = cx as i32;
    let cy = cy as i32;
    let radius = radius as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 {
                    // We ignore out of bound pixels
                    continue;
                }
                if !valid_px.contains(&(x as u32, y as u32)) {
                    return false;
                }
            }
        }
    }

    true
}
