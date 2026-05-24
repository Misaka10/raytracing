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
    #[arg(long, default_value = "600")]
    width: u32,

    #[arg(long, default_value = "100")]
    samples: u32,

    #[arg(long, default_value = "50")]
    max_depth: u32,

    #[arg(long, default_value = "output.png")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

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
    cam.aspect_ratio = 1.0;
    cam.image_width = args.width;
    cam.samples_per_pixel = args.samples;
    cam.max_depth = args.max_depth;
    cam.background = Color::zero();
    cam.vfov = 40.0;
    cam.lookfrom = Point3::new(278.0, 278.0, -800.0);
    cam.lookat = Point3::new(278.0, 278.0, 0.0);
    cam.vup = Vec3::new(0.0, 1.0, 0.0);
    cam.defocus_angle = 0.0;
    cam.initialize();

    let world_hittable = Hittable::HittableList(world.clone());
    let lights_hittable = Hittable::HittableList(lights);

    cam.render(&world_hittable, &lights_hittable, &args.output)?;

    Ok(())
}
