use bpaf::{Parser, construct, positional, short};
use image::{GenericImageView, ImageBuffer, Rgb, RgbImage};
use opencv::{
    core::{CV_8UC1, CV_32F, Mat, MatTrait, Point, Scalar, min_max_loc, no_array},
    imgproc,
};
use std::{collections::BTreeSet, fmt::Display, path::PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
enum BadError {
    #[error("error thrown by OpenCV")]
    OpenCVError(#[from] opencv::Error),
}

const MAX_RADIUS: i32 = 25;

#[derive(Clone, Debug, Hash)]
struct Pixel {
    x: i32,
    y: i32,
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
    x: i32,
    y: i32,
    r: i32,
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

// Compute the Euclidian Distance Transform using OpenCV.
fn compute_edt(mask: &BTreeSet<Pixel>, width: i32, height: i32) -> Result<Pixel, BadError> {
    let mut binary_image =
        Mat::new_rows_cols_with_default(height, width, CV_8UC1, Scalar::all(0.0)).unwrap();
    for px in mask.iter() {
        if let Ok(pixel) = binary_image.at_2d_mut::<u8>(px.y, px.x) {
            *pixel = 255;
        }
    }
    let mut dist = Mat::default();
    imgproc::distance_transform(
        &binary_image,
        &mut dist,
        imgproc::DIST_L2,
        imgproc::DIST_MASK_PRECISE,
        CV_32F,
    )
    .map_err(|e| BadError::OpenCVError(e))?;

    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    min_max_loc(
        &dist,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &no_array(),
    )
    .map_err(|e| BadError::OpenCVError(e))?;
    Ok(Pixel {
        x: max_loc.x,
        y: max_loc.y,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    // Using many core doesn't help for EDT and slow down the start.
    // Plus parrallelism is handle by converting many frames.
    opencv::core::set_num_threads(1)?;
    let img = image::open(&args.path)?;
    let width: i32 = img.width() as i32;
    let height: i32 = img.height() as i32;
    let targets_pixels: BTreeSet<Pixel> = (0..(width * height))
        .map(|d1| Pixel {
            x: d1 % width,
            y: d1 / width,
        })
        .filter(|coord| {
            let pixel = img.get_pixel(coord.x as u32, coord.y as u32).0;
            let luminance = calculate_luminance(pixel[0], pixel[1], pixel[2]);
            return luminance > 0.5;
        })
        .collect();

    if args.debug {
        debug_img(&targets_pixels, width, height, "targets.png")?;
    }

    let mut circles: Vec<Circle> = Vec::new();
    let mut mask = targets_pixels.clone();
    while mask.len() > 0 {
        match compute_edt(&mask, width, height) {
            Ok(max_point) => {
                let circle = find_biggest_circle(&mask, max_point.x, max_point.y, MAX_RADIUS);
                if circle.r < 2 {
                    mask.remove(&Pixel {
                        x: max_point.x,
                        y: max_point.y,
                    });
                    continue;
                }
                circles.push(circle.clone());
                let to_rm = pixels_in_circle(circle.x, circle.y, circle.r);
                for px in to_rm {
                    mask.remove(&px);
                }
            }
            Err(e) => {
                eprintln!("Got an error while computing the edt: {}", e);
                break;
            }
        }
    }

    if args.debug {
        let circle_px: BTreeSet<Pixel> = circles
            .iter()
            .map(|e| pixels_in_circle(e.x, e.y, e.r))
            .reduce(|mut acc, e| {
                acc.extend(e);
                return acc;
            })
            .unwrap()
            .into_iter()
            .collect();
        debug_img(&circle_px, width, height, "output.png")?;
    }

    // Output the circles.
    let circle_fmt = circles
        .iter()
        .map(|c| Circle {
            y: (height - c.y), // In echart, y=0 is at the bottom
            x: c.x,
            r: c.r,
        })
        .map(|c| c.to_string())
        .collect::<Vec<String>>()
        .join(",");

    let arr = format!("[{}],", circle_fmt);
    println!("{}", arr);

    Ok(())
}

fn find_biggest_circle(valid_px: &BTreeSet<Pixel>, cx: i32, cy: i32, max_radius: i32) -> Circle {
    let mut circle = Circle { x: cx, y: cy, r: 1 };
    for r in 2..=max_radius {
        let valid = check_circle(valid_px, cx, cy, r);
        if !valid {
            break;
        }
        circle.r = r;
    }
    circle
}

fn pixels_in_circle(cx: i32, cy: i32, r: i32) -> Vec<Pixel> {
    let mut pixels = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 {
                    // We ignore out of bound pixels
                    continue;
                }
                pixels.push(Pixel { x, y });
            }
        }
    }
    pixels
}

fn check_circle(valid_px: &BTreeSet<Pixel>, cx: i32, cy: i32, radius: i32) -> bool {
    let radius = radius as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 {
                    return false;
                }
                let px = Pixel { x, y };
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
    pixel_set: &BTreeSet<Pixel>,
    width: i32,
    height: i32,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]); // Black color
    }
    for p in pixel_set {
        if p.x < width && p.y < height {
            img.put_pixel(p.x as u32, p.y as u32, Rgb([255, 255, 255]));
        }
    }
    img.save(filename)?;
    Ok(())
}
