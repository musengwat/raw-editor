use rawler::{RawImageData, decode_file};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = decode_file(&a[1]).unwrap();
    let RawImageData::Integer(d) = &m.data else { panic!() };
    let crop = m.crop_area.unwrap();
    let (x0,y0,x1,y1): (usize,usize,usize,usize) = (a[2].parse().unwrap(),a[3].parse().unwrap(),a[4].parse().unwrap(),a[5].parse().unwrap());
    let mut v=[vec![],vec![],vec![]];
    for y in y0..y1 { for x in x0..x1 { let i=(y+crop.p.y)*m.width+x+crop.p.x; for c in 0..3 { v[c].push(d[i*3+c]); } } }
    for c in 0..3 { v[c].sort_unstable(); let n=v[c].len(); println!("ch{c}: p5 {} p50 {} p95 {} max {}", v[c][n/20], v[c][n/2], v[c][n*19/20], v[c][n-1]); }
    // global top values per channel
    for c in 0..3 { let mut h=std::collections::HashMap::new(); for y in 0..crop.d.h { for x in 0..crop.d.w { let i=(y+crop.p.y)*m.width+x+crop.p.x; *h.entry(d[i*3+c]).or_insert(0u32)+=1; } }
      let mut t: Vec<_> = h.into_iter().filter(|(k,_)| *k>8000).collect(); t.sort_by_key(|x| std::cmp::Reverse(x.1)); println!("ch{c} top: {:?}", &t[..5.min(t.len())]); }
}
