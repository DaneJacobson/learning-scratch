use std::mem;
use std::arch::aarch64::*;
use std::f64::consts::PI;

unsafe fn offset_Momentum(bodies: *mut body) {
    for i in 0..BODIES_COUNT {
        for m in 0..3 {
            (*bodies.add(0)).velocity[m] -=
                (*bodies.add(i)).velocity[m] 
                * (*bodies.add(i)).mass / SOLAR_MASS;
        }
    }
}

unsafe fn output_Energy(bodies: *mut body){
    let mut energy = 0.;
    for i in 0..BODIES_COUNT {
        // Add the kinetic energy for each body.
        energy += 0.5 * (*bodies.add(i)).mass * (
            (*bodies.add(i)).velocity[0]
                * (*bodies.add(i)).velocity[0]
            + (*bodies.add(i)).velocity[1]
                * (*bodies.add(i)).velocity[1]
            + (*bodies.add(i)).velocity[2]
                * (*bodies.add(i)).velocity[2]);

        // Add the potential energy between this body and
        // every other body.
        for j in i+1..BODIES_COUNT {
            let mut position_Delta =   // <----------- Note 1
                [mem::MaybeUninit::<f64>::uninit(); 3];
            for m in 0..3 {
                position_Delta[m].as_mut_ptr().write(
                    (*bodies.add(i)).position[m]
                        - (*bodies.add(j)).position[m]
                );
            }
            let position_Delta: [f64; 3] = // <------- Note 2
                mem::transmute(position_Delta);

            energy -= (*bodies.add(i)).mass
                * (*bodies.add(j)).mass
                / f64::sqrt(               // <------- Note 3
                      position_Delta[0]*position_Delta[0]+
                      position_Delta[1]*position_Delta[1]+
                      position_Delta[2]*position_Delta[2]
                  );
        }
    }

    // Output the total energy of the system.
    println!("{:.9}", energy);
}

fn main() {
    #![allow(
        non_upper_case_globals,
        non_camel_case_types,
        non_snake_case,
    )]

    #[repr(C)]  
        struct body {
        position: [f64; 3],
        velocity: [f64; 3],
        mass: f64,
    }    
    println!("{}", PI);
}
