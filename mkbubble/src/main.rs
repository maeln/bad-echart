use bpaf::{Parser, construct, positional, short};
use image::{GenericImageView, ImageBuffer, Rgb, RgbImage};
use indexmap::set::IndexSet;
use opencv::{
    core::{CV_8UC1, CV_32F, Mat, MatTrait, Point, Scalar, min_max_loc, no_array},
    imgproc,
};
use std::{
    collections::{BTreeSet, HashSet},
    fmt::Display,
    path::PathBuf,
    sync::Arc,
};

const MAX_RADIUS: u32 = 25;

#[derive(Clone, Debug, Hash)]
struct Pixel {
    x: u32,
    y: u32,
}

impl PartialEq for Pixel {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}
impl PartialOrd for Pixel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.x.partial_cmp(&other.x) {
            Some(std::cmp::Ordering::Equal) => self.y.partial_cmp(&other.y),
            other => other,
        }
    }
}
impl Eq for Pixel {}
impl Ord for Pixel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.x.cmp(&other.x) {
            std::cmp::Ordering::Equal => self.y.cmp(&other.y),
            other => other,
        }
    }
}

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

struct Args {
    path: PathBuf,
    debug: bool,
}

fn parse_args() -> Args {
    let path = positional::<PathBuf>("IMG").help("frame to analyze");
    let debug = short('d')
        .long("debug")
        .help("activate debug mode: output the mask and pass images.")
        .switch();
    construct!(Args { path, debug }).to_options().run()
}

fn compute_edt(mask: &IndexSet<Pixel>, width: i32, height: i32) -> Option<Pixel> {
    let mut binary_image =
        Mat::new_rows_cols_with_default(height, width, CV_8UC1, Scalar::all(0.0)).unwrap();
    for px in mask.iter() {
        if let Ok(pixel) = binary_image.at_2d_mut::<u8>(px.y as i32, px.x as i32) {
            *pixel = 255;
        }
    }
    let mut dist = Mat::default();
    if let Err(e) = imgproc::distance_transform(
        &binary_image,
        &mut dist,
        imgproc::DIST_L2,
        imgproc::DIST_MASK_PRECISE,
        CV_32F,
    ) {
        eprintln!("could not compute edt: {}", e);
    }

    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    if let Err(e) = min_max_loc(
        &dist,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &no_array(),
    ) {
        eprintln!("could not find min/max in edt: {}", e);
    }

    Some(Pixel {
        x: max_loc.x as u32,
        y: max_loc.y as u32,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    opencv::core::set_num_threads(1)?;
    let img = image::open(&args.path)?;
    let targets_pixels: IndexSet<Pixel> = (0..(img.width() * img.height()))
        .map(|d1| Pixel {
            x: d1 % img.width(),
            y: d1 / img.width(),
        })
        .filter(|coord| {
            let pixel = img.get_pixel(coord.x, coord.y).0;
            let luminance = calculate_luminance(pixel[0], pixel[1], pixel[2]);
            return luminance > 0.5;
        })
        .collect();

    if args.debug {
        debug_img(&targets_pixels, img.width(), img.height(), "targets.png")
            .expect("could not save debug img");
    }

    let mut circles: Vec<Circle> = Vec::new();
    let mut mask = targets_pixels.clone();
    // let mut all_pixels_for_transform = Vec::with_capacity((img.width() * img.height()) as usize);
    // let mut max_v: Option<u32> = None;

    while mask.len() > 0 {
        if let Some(max_point) = compute_edt(&mask, img.width() as i32, img.height() as i32) {
            let circle = find_biggest_circle(&mask, max_point.x, max_point.y, MAX_RADIUS);
            if circle.r < 3 {
                mask.swap_remove(&Pixel {
                    x: max_point.x,
                    y: max_point.y,
                });
                continue;
            }
            circles.push(circle.clone());

            // 5. Remove the pixels of this new circle from the mask and repeat
            let to_rm = pixels_in_circle(circle.x, circle.y, circle.r);
            for px in to_rm {
                mask.swap_remove(&px);
            }
        } else {
            break;
        }
    }

    if args.debug {
        let circle_px: IndexSet<Pixel> = circles
            .iter()
            .map(|e| pixels_in_circle(e.x, e.y, e.r))
            .reduce(|mut acc, e| {
                acc.extend(e);
                return acc;
            })
            .unwrap()
            .into_iter()
            .collect();
        debug_img(&circle_px, img.width(), img.height(), "fist_pass.png")
            .expect("could not save debug img");
    }

    // Second pass: We make a new image from the first with the circles
    // of the first pass removed, but we make them a bit smaller to allow for some overlap
    // and we again find circles
    // for circle in circles.iter() {
    //     pixels_in_circle(circle.x, circle.y, circle.r)
    //         .iter()
    //         .for_each(|p| {
    //             second_pass_mask.remove(p);
    //         });
    // }
    // if args.debug {
    //     debug_img(&mask, img.width(), img.height(), "mask.bmp").expect("could not save debug img");
    // }
    // while second_pass_pixels.len() > 0 {
    //     let coord = second_pass_pixels.iter().next().unwrap().clone();
    //     second_pass_pixels.remove(&coord);
    //     let circle = find_biggest_circle(&mask, coord.0, coord.1, MAX_RADIUS);
    //     if circle.r < 2 {
    //         continue;
    //     }
    //     let to_rm = pixels_in_circle(circle.x, circle.y, circle.r);
    //     circles.push(circle);
    //     for px in to_rm {
    //         mask.remove(&px);
    //     }
    // }

    if args.debug {
        let circle_px: IndexSet<Pixel> = circles
            .iter()
            .map(|e| pixels_in_circle(e.x, e.y, e.r))
            .reduce(|mut acc, e| {
                acc.extend(e);
                return acc;
            })
            .unwrap()
            .into_iter()
            .collect();
        debug_img(&circle_px, img.width(), img.height(), "output.png")
            .expect("could not save debug img");
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

fn find_biggest_circle(valid_px: &IndexSet<Pixel>, cx: u32, cy: u32, max_radius: u32) -> Circle {
    let mut circle = Circle { x: cx, y: cy, r: 1 };
    for r in 3..=max_radius {
        let valid = check_circle(valid_px, cx, cy, r);
        // eprintln!("val: {} {}", valid, r);
        if !valid {
            break;
        }
        circle.r = r;
    }
    circle
}

fn pixels_in_circle(cx: u32, cy: u32, r: u32) -> Vec<Pixel> {
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
                pixels.push(Pixel {
                    x: x as u32,
                    y: y as u32,
                });
            }
        }
    }
    pixels
}

fn check_circle(valid_px: &IndexSet<Pixel>, cx: u32, cy: u32, radius: u32) -> bool {
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
                let px = Pixel {
                    x: x as u32,
                    y: y as u32,
                };
                if !valid_px.contains(&px) {
                    return false;
                }
            }
        }
    }

    true
}

fn calculate_luminance(r: u8, g: u8, b: u8) -> f32 {
    // Using the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    let r_norm = r as f32 / 255.0;
    let g_norm = g as f32 / 255.0;
    let b_norm = b as f32 / 255.0;
    0.299 * r_norm + 0.587 * g_norm + 0.114 * b_norm
}

fn debug_img(
    pixel_set: &IndexSet<Pixel>,
    width: u32,
    height: u32,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img: RgbImage = ImageBuffer::new(width, height);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]); // Black color
    }
    for p in pixel_set {
        if p.x < width && p.y < height {
            img.put_pixel(p.x, p.y, Rgb([255, 255, 255]));
        }
    }
    img.save(filename)?;
    Ok(())
}
