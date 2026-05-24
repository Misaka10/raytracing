use clap::Parser;
use rt_next_week::camera::Camera;
use rt_next_week::material::Material;
use rt_next_week::quad::Quad;
use rt_next_week::quad_box;
use rt_next_week::sphere::Sphere;
use rt_next_week::vec3::{Color, Point3, Vec3};
use rt_next_week::{Hittable, HittableList};

#[derive(Parser)]
#[command(name = "rt-next-week")]
#[command(about = "Physically based Monte Carlo path tracer (Rust port)")]
struct Args {
    #[arg(long, default_value = "3840")]
    width: u32,

    /// 图像高度（0 = 从宽高比自动推导）
    #[arg(long, default_value = "2160")]
    height: u32,

    #[arg(long, default_value = "1.777")]
    aspect_ratio: f64,

    #[arg(long, default_value = "400")]
    samples: u32,

    #[arg(long, default_value = "75")]
    max_depth: u32,

    #[arg(long, default_value = "output.png")]
    output: String,

    /// 随机种子（可选，提供则确定性渲染，同 seed 同参数输出一致）
    #[arg(long)]
    seed: Option<u64>,

    /// 启用 JSON 进度输出（用于 Electron IPC）
    #[arg(long, default_value_t = false)]
    json: bool,

    /// 使用 GPU (OptiX RT Core) 渲染
    #[arg(long, default_value_t = false)]
    gpu: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 初始化 rayon 线程池，增大栈空间防止递归 ray_color 爆栈
    rayon::ThreadPoolBuilder::new()
        .stack_size(16 * 1024 * 1024)
        .build_global()
        .unwrap_or_else(|_| {});

    let mut world = HittableList::new();

    let red = Material::lambertian_color(Color::new(0.65, 0.05, 0.05));
    let white = Material::lambertian_color(Color::new(0.73, 0.73, 0.73));
    let green = Material::lambertian_color(Color::new(0.12, 0.45, 0.15));
    let light = Material::diffuse_light_color(Color::new(15.0, 15.0, 15.0));

    // Cornell box sides
    world.add(Hittable::Quad(Quad::new(
        Point3::new(555.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 555.0), Vec3::new(0.0, 555.0, 0.0), green,
    )));
    world.add(Hittable::Quad(Quad::new(
        Point3::new(0.0, 0.0, 555.0), Vec3::new(0.0, 0.0, -555.0), Vec3::new(0.0, 555.0, 0.0), red,
    )));
    world.add(Hittable::Quad(Quad::new(
        Point3::new(0.0, 555.0, 0.0), Vec3::new(555.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 555.0), white.clone(),
    )));
    world.add(Hittable::Quad(Quad::new(
        Point3::new(0.0, 0.0, 555.0), Vec3::new(555.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -555.0), white.clone(),
    )));
    world.add(Hittable::Quad(Quad::new(
        Point3::new(555.0, 0.0, 555.0), Vec3::new(-555.0, 0.0, 0.0), Vec3::new(0.0, 555.0, 0.0), white.clone(),
    )));

    // Light
    world.add(Hittable::Quad(Quad::new(
        Point3::new(213.0, 554.0, 227.0), Vec3::new(130.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 105.0), light,
    )));

    // Box
    let box_geom = quad_box::make_box(
        &Point3::new(0.0, 0.0, 0.0), &Point3::new(165.0, 330.0, 165.0), white.clone(),
    );
    let box_rotated = Hittable::rotate_y(box_geom, 15.0);
    let box_translated = Hittable::translate(box_rotated, Vec3::new(265.0, 0.0, 295.0));
    world.add(box_translated);

    // Glass sphere
    let glass = Material::dielectric(1.5);
    world.add(Hittable::Sphere(Sphere::stationary(
        Point3::new(190.0, 90.0, 190.0), 90.0, glass,
    )));

    // Light sources for importance sampling
    let mut lights = HittableList::new();
    let empty_mat = Material::lambertian_color(Color::zero());
    lights.add(Hittable::Quad(Quad::new(
        Point3::new(343.0, 554.0, 332.0), Vec3::new(-130.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -105.0), empty_mat.clone(),
    )));
    lights.add(Hittable::Sphere(Sphere::stationary(
        Point3::new(190.0, 90.0, 190.0), 90.0, empty_mat,
    )));

    let mut cam = Camera::new();
    cam.aspect_ratio = args.aspect_ratio;
    cam.image_width = args.width;
    cam.image_height = args.height;
    cam.samples_per_pixel = args.samples;
    cam.max_depth = args.max_depth;
    cam.background = Color::zero();
    cam.vfov = 40.0;
    cam.lookfrom = Point3::new(278.0, 278.0, -800.0);
    cam.lookat = Point3::new(278.0, 278.0, 0.0);
    cam.vup = Vec3::new(0.0, 1.0, 0.0);
    cam.defocus_angle = 0.0;
    cam.initialize();

    // 使用 BVH 加速几何体命中测试（O(log n) 替代 O(n)）
    let mut objects = world.objects;
    let bvh = rt_next_week::bvh::BvhNode::from_objects(&mut objects);
    let world_hittable = Hittable::BvhNode(bvh);
    let lights_hittable = Hittable::HittableList(lights);

    if args.gpu {
        #[cfg(feature = "cuda")]
        cam.render_gpu(&world_hittable, &args.output, args.seed)?;
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("GPU support requires --features cuda. Rebuild with: cargo build --release --features cuda");
    } else {
        cam.render(&world_hittable, &lights_hittable, &args.output, args.seed, args.json)?;
    }

    Ok(())
}
