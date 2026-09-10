use minifb::{Key, Window, WindowOptions};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use ultralytics_inference::{
    model::YOLOModel,
    Device,
    InferenceConfig,
};

use v4l::{
    buffer::Type,
    io::traits::CaptureStream,
    video::Capture,
    Device as V4lDevice,
    FourCC,
};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const MODEL_SIZE: usize = 768;
const SKIP: usize = 3;

// ============================================================
// YUYV -> RGB
// ============================================================

fn yuyv_to_rgb(
    data: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];

    for i in 0..(width * height / 2) {
        let src = i * 4;
        let dst = i * 6;

        let y0 = data[src] as f32;
        let u  = data[src + 1] as f32;
        let y1 = data[src + 2] as f32;
        let v  = data[src + 3] as f32;

        let c0 = y0 - 16.0;
        let c1 = y1 - 16.0;
        let d = u - 128.0;
        let e = v - 128.0;

        rgb[dst] =
            (1.164 * c0 + 1.596 * e)
                .clamp(0.0, 255.0) as u8;

        rgb[dst + 1] =
            (1.164 * c0 - 0.392 * d - 0.813 * e)
                .clamp(0.0, 255.0) as u8;

        rgb[dst + 2] =
            (1.164 * c0 + 2.017 * d)
                .clamp(0.0, 255.0) as u8;

        rgb[dst + 3] =
            (1.164 * c1 + 1.596 * e)
                .clamp(0.0, 255.0) as u8;

        rgb[dst + 4] =
            (1.164 * c1 - 0.392 * d - 0.813 * e)
                .clamp(0.0, 255.0) as u8;

        rgb[dst + 5] =
            (1.164 * c1 + 2.017 * d)
                .clamp(0.0, 255.0) as u8;
    }

    rgb
}


// ============================================================
// Latest RGB frame
// ============================================================

type LatestFrame = Arc<Mutex<Option<Vec<u8>>>>;


// ============================================================
// Depth frame
// ============================================================

struct DepthFrame {
    data: Vec<f32>,
    width: usize,
    height: usize,
    min_depth: f32,
    max_depth: f32,
}

type LatestDepth = Arc<Mutex<Option<Arc<DepthFrame>>>>;


// ============================================================
// Camera thread
// ============================================================

fn camera_thread(latest_frame: LatestFrame) {
    let mut dev = match V4lDevice::new(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("Camera open error: {err}");
            return;
        }
    };

    let mut fmt = match dev.format() {
        Ok(fmt) => fmt,
        Err(err) => {
            eprintln!("Camera format error: {err}");
            return;
        }
    };

    fmt.width = WIDTH as u32;
    fmt.height = HEIGHT as u32;
    fmt.fourcc = FourCC::new(b"YUYV");

    if let Err(err) = dev.set_format(&fmt) {
        eprintln!("Camera set format error: {err}");
        return;
    }

    let mut stream =
        match v4l::io::mmap::Stream::with_buffers(
            &mut dev,
            Type::VideoCapture,
            4,
        ) {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("Camera stream error: {err}");
                return;
            }
        };

    println!("camera thread");
    println!("YUYV {}x{} @ 30 FPS", WIDTH, HEIGHT);
    let mut skip: usize = 0;
    loop {

        let (data, _) = match stream.next() {
            Ok(frame) => frame,
            Err(err) => {
                eprintln!("Camera frame error: {err}");
                continue;
            }
        };

        if skip < SKIP {
            skip += 1;
            continue;
        }
        skip = 0;
        

        let rgb = yuyv_to_rgb(
            data,
            WIDTH,
            HEIGHT,
        );

        // ----------------------------------------------------
        // 关键：
        // 只保存最新 frame
        //
        // 如果 GPU 还在处理上一帧，
        // 这里会直接覆盖旧 frame。
        // ----------------------------------------------------

        if let Ok(mut slot) = latest_frame.lock() {
            *slot = Some(rgb);
        }
    }
}


// ============================================================
// Inference thread
// ============================================================

fn inference_thread(
    latest_frame: LatestFrame,
    latest_depth: LatestDepth,
) {
    // --------------------------------------------------------
    // Load YOLO model
    // --------------------------------------------------------

    let config = InferenceConfig::new()
        .with_confidence(0.45)
        .with_iou(0.45)
        .with_imgsz(MODEL_SIZE, MODEL_SIZE)
        .with_device(Device::Cuda(0));

    println!("Loading CUDA depth model...");

    let mut model = match YOLOModel::load_with_config(
        "./models/yolo26s-depth.onnx",
        config,
    ) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("Model load error: {err}");
            return;
        }
    };

    println!("CUDA depth model ready.");
    println!("Inference thread started.");

    loop {
        // ----------------------------------------------------
        // Take latest frame
        // ----------------------------------------------------

        let rgb = {
            let mut slot = match latest_frame.lock() {
                Ok(slot) => slot,
                Err(_) => return,
            };

            slot.take()
        };

        let rgb = match rgb {
            Some(rgb) => rgb,

            None => {
                thread::sleep(
                    Duration::from_millis(1)
                );
                continue;
            }
        };

        // ----------------------------------------------------
        // RGB buffer -> DynamicImage
        // ----------------------------------------------------

        let rgb_image =
            match image::RgbImage::from_raw(
                WIDTH as u32,
                HEIGHT as u32,
                rgb,
            ) {
                Some(img) => img,

                None => {
                    eprintln!("Invalid RGB buffer");
                    continue;
                }
            };

        let img =
            image::DynamicImage::ImageRgb8(
                rgb_image
            );

        // ----------------------------------------------------
        // YOLO inference
        // ----------------------------------------------------

        let t = std::time::Instant::now();
        let results = match model.predict_image(
            &img,
            "camera".to_string(),
        ) {
            Ok(results) => results,

            Err(err) => {
                eprintln!("Inference error: {err}");
                continue;
            }
        };
        println!("predict: {:?}", t.elapsed());

        // ----------------------------------------------------
        // Get depth map
        // ----------------------------------------------------

        let depth =
            match results
                .first()
                .and_then(|r| r.depth.as_ref())
            {
                Some(depth) => depth,

                None => {
                    eprintln!("No depth map returned");
                    continue;
                }
            };

        let depth_width =
            depth.data.shape()[1];

        let depth_height =
            depth.data.shape()[0];

        let min_depth =
            depth.min_depth()
                .unwrap_or(0.0);

        let max_depth =
            depth.max_depth()
                .unwrap_or(1.0);

        // ----------------------------------------------------
        // Copy depth data out of YOLO result
        // ----------------------------------------------------

        let depth_data =
            depth.data.iter()
                .copied()
                .collect::<Vec<f32>>();

        let depth_frame = Arc::new(
            DepthFrame {
                data: depth_data,
                width: depth_width,
                height: depth_height,
                min_depth,
                max_depth,
            }
        );

        // ----------------------------------------------------
        // Replace old depth frame
        //
        // 只保留最新结果
        // ----------------------------------------------------

        if let Ok(mut slot) =
            latest_depth.lock()
        {
            *slot = Some(depth_frame);
        }
    }
}


// ============================================================
// Main
// ============================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // --------------------------------------------------------
    // Shared latest frame
    // --------------------------------------------------------

    let latest_frame: LatestFrame =
        Arc::new(Mutex::new(None));

    // --------------------------------------------------------
    // Shared latest depth
    // --------------------------------------------------------

    let latest_depth: LatestDepth =
        Arc::new(Mutex::new(None));

    // --------------------------------------------------------
    // Camera thread
    // --------------------------------------------------------

    {
        let latest_frame =
            Arc::clone(&latest_frame);

        let _ = thread::Builder::new()
            .name("yolo26-camera".into())
            .spawn(move || {
                camera_thread(latest_frame);
            });
    }

    // --------------------------------------------------------
    // Inference thread
    // --------------------------------------------------------

    {
        let latest_frame =
            Arc::clone(&latest_frame);

        let latest_depth =
            Arc::clone(&latest_depth);

        let _ = thread::Builder::new()
            .name("yolo26-inference".into())
            .spawn(move || {
                inference_thread(
                    latest_frame,
                    latest_depth,
                );
            });
    }

    // --------------------------------------------------------
    // Display window
    // --------------------------------------------------------

    let mut window = Window::new(
        "YOLO26 Depth - NVIDIA T500",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )?;

    window.set_target_fps(30);

    let mut framebuffer =
        vec![0u32; WIDTH * HEIGHT];

    println!("YOLO26 Depth started.");
    println!("Latest-frame pipeline enabled.");
    println!("Press ESC to exit.");

    // --------------------------------------------------------
    // Display loop
    // --------------------------------------------------------

    while window.is_open()
        && !window.is_key_down(Key::Escape)
    {
        // ----------------------------------------------------
        // Get latest depth
        // ----------------------------------------------------

        let depth =
            match latest_depth.lock() {
                Ok(slot) => slot.as_ref().cloned(),
                Err(_) => None,
            };

        if let Some(depth) = depth {

            let range =
                (depth.max_depth
                    - depth.min_depth)
                    .max(0.001);

            // ------------------------------------------------
            // Depth -> grayscale
            // ------------------------------------------------

            for y in 0..HEIGHT {
                for x in 0..WIDTH {

                    let dx =
                        x * depth.width / WIDTH;

                    let dy =
                        y * depth.height / HEIGHT;

                    let value =
                        depth.data[
                            dy * depth.width + dx
                        ];

                    let normalized =
                        if value > 0.0 {
                            ((value
                                - depth.min_depth)
                                / range)
                                .clamp(0.0, 1.0)
                        } else {
                            0.0
                        };

                    let gray =
                        (normalized * 255.0)
                            as u32;

                    framebuffer[
                        y * WIDTH + x
                    ] =
                        (gray << 16)
                        | (gray << 8)
                        | gray;
                }
            }
        }

        // ----------------------------------------------------
        // Display
        // ----------------------------------------------------

        window.update_with_buffer(
            &framebuffer,
            WIDTH,
            HEIGHT,
        )?;
    }

    Ok(())
}