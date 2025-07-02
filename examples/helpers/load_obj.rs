use std::{fs::File, io::BufReader, path::Path};

use obj::{LoadConfig, ObjData};
use obvhs::triangle::Triangle;
use ruzstd::decoding::StreamingDecoder;

pub fn load_obj_mesh_data(model_path: &Path) -> Vec<Vec<Triangle>> {
    let file = File::open(model_path).unwrap();
    let obj_data = if model_path.extension().unwrap().to_str().unwrap() == "zst" {
        ObjData::load_buf_with_config(
            BufReader::new(StreamingDecoder::new(BufReader::new(file)).unwrap()),
            LoadConfig::default(),
        )
        .unwrap()
    } else {
        ObjData::load_buf_with_config(BufReader::new(file), LoadConfig::default()).unwrap()
    };

    let mut objects = Vec::with_capacity(obj_data.objects.len());
    for obj in obj_data.objects {
        let mut triangles = Vec::new();
        for group in obj.groups {
            for poly in group.polys {
                let a = obj_data.position[poly.0[0].0].into();
                let b = obj_data.position[poly.0[1].0].into();
                let c = obj_data.position[poly.0[2].0].into();
                triangles.push(Triangle {
                    v0: a,
                    v1: b,
                    v2: c,
                });
                if poly.0.len() == 4 {
                    let d = obj_data.position[poly.0[3].0].into();
                    triangles.push(Triangle {
                        v0: a,
                        v1: c,
                        v2: d,
                    });
                }
            }
        }
        objects.push(triangles);
    }
    objects
}
