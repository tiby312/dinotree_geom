use crate::axgeom::*;
use bit_vec::*;

///Represents one of 8 cardinal directions (with diagonals).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CardDir2 {
    UU = 0,
    UR = 1,
    RR = 2,
    RD = 3,
    DD = 4,
    DL = 5,
    LL = 6,
    UL = 7,
}
impl CardDir2 {
    pub fn from_u8(a: u8) -> CardDir2 {
        use CardDir2::*;
        match a {
            0 => UU,
            1 => UR,
            2 => RR,
            3 => RD,
            4 => DD,
            5 => DL,
            6 => LL,
            7 => UL,
            _ => panic!("{:?} is not a valid value", a),
        }
    }
    pub fn from_offset(offset: Vec2<GridNum>) -> CardDir2 {
        use CardDir2::*;
        match offset {
            Vec2 { x: 0, y: -1 } => UU,
            Vec2 { x: 1, y: -1 } => UR,
            Vec2 { x: 1, y: 0 } => RR,
            Vec2 { x: 1, y: 1 } => RD,
            Vec2 { x: 0, y: 1 } => DD,
            Vec2 { x: -1, y: 1 } => DL,
            Vec2 { x: -1, y: 0 } => LL,
            Vec2 { x: -1, y: -1 } => UL,
            _ => {
                unreachable!("Invalid offset provided: {:?}", offset);
            }
        }
    }

    pub fn all_offsets() -> [(Vec2<GridNum>, usize); 8] {
        [
            CardDir2::UU.into_offset(),
            CardDir2::UR.into_offset(),
            CardDir2::RR.into_offset(),
            CardDir2::RD.into_offset(),
            CardDir2::DD.into_offset(),
            CardDir2::DL.into_offset(),
            CardDir2::LL.into_offset(),
            CardDir2::UL.into_offset(),
        ]
    }

    pub fn into_offset(self) -> (Vec2<GridNum>, usize) {
        match self {
            CardDir2::UU => (vec2(0, -1), 2),
            CardDir2::UR => (vec2(1, -1), 3),
            CardDir2::RR => (vec2(1, 0), 2),
            CardDir2::RD => (vec2(1, 1), 3),
            CardDir2::DD => (vec2(0, 1), 2),
            CardDir2::DL => (vec2(-1, 1), 3),
            CardDir2::LL => (vec2(-1, 0), 2),
            CardDir2::UL => (vec2(-1, -1), 3),
        }
    }
}

use serde::{Deserialize, Serialize};

///Represents one of 4 carinal directions (without diagonals)
#[derive(Hash, Serialize, Deserialize, Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]

pub enum CardDir {
    U,
    D,
    L,
    R,
}
impl CardDir {
    pub fn into_char(self) -> char {
        use CardDir::*;
        match self {
            U => '↑',
            D => '↓',
            L => '←',
            R => '→',
        }
    }
    pub fn into_vec(self) -> Vec2<GridNum> {
        use CardDir::*;
        match self {
            U => vec2(0, -1),
            D => vec2(0, 1),
            L => vec2(-1, 0),
            R => vec2(1, 0),
        }
    }
    pub fn into_two_bits(self) -> u8 {
        use CardDir::*;
        match self {
            U => 0b00,
            D => 0b01,
            L => 0b10,
            R => 0b11,
        }
    }
}

///Iterate over every cell position in a grid.
#[derive(Copy, Clone)]
pub struct Iterator2D {
    counter: Vec2<GridNum>,
    dim: Vec2<GridNum>,
}
impl Iterator2D {
    pub fn new(dim: Vec2<GridNum>) -> Iterator2D {
        Iterator2D {
            counter: vec2same(0),
            dim,
        }
    }
}

use core::iter::*;

impl FusedIterator for Iterator2D {}
impl ExactSizeIterator for Iterator2D {}
impl Iterator for Iterator2D {
    type Item = Vec2<GridNum>;
    fn size_hint(&self) -> (usize, Option<usize>) {
        let diff = vec2(self.dim.x, self.dim.y - 1) - self.counter;
        //TODO test this
        let l = (self.dim.x * diff.y + diff.x) as usize;
        (l, Some(l))
    }
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter.y == self.dim.y {
            return None;
        }

        let k = self.counter;

        self.counter.x += 1;
        if self.counter.x == self.dim.x {
            self.counter.x = 0;
            self.counter.y += 1;
        }
        Some(k)
    }
}

#[test]
fn test_iterator2d() {
    let i = Iterator2D::new(vec2(10, 20));
    assert_eq!(i.len(), 200);
    assert_eq!(i.count(), 200);

    let i = Iterator2D::new(vec2(20, 10));
    assert_eq!(i.len(), 200);
    assert_eq!(i.count(), 200);
}

impl<'a> FusedIterator for CellIterator<'a> {}
impl<'a> ExactSizeIterator for CellIterator<'a> {}

///Iterate over every element in a grid
pub struct CellIterator<'a> {
    grid: &'a Grid2D,
    inner: Iterator2D,
}
impl<'a> Iterator for CellIterator<'a> {
    type Item = (Vec2<GridNum>, bool);

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(v) => Some((v, self.grid.get(v))),
            None => None,
        }
    }
}

///Represents a ascii map where █ are treated as walls and spaces are treated as empty cells.
///This is still in its string representation.
pub struct Map<'a> {
    pub dim: Vec2<GridNum>,
    pub str: &'a str,
}

///The type used for the dimension and indexing of the grid
pub type GridNum = i16;

pub struct GridByte2D<T> {
    dim: Vec2<GridNum>,
    inner: Vec<T>,
}
impl<T: Copy> GridByte2D<T> {
    pub fn new(dim: Vec2<GridNum>, val: T) -> GridByte2D<T> {
        let mut inner = Vec::new();
        inner.resize((dim.x * dim.y) as usize, val);
        GridByte2D { dim, inner }
    }
}

impl<T> GridByte2D<T> {
    pub fn dim(&self) -> Vec2<GridNum> {
        self.dim
    }

    pub fn contains(&self, point: Vec2<GridNum>) -> bool {
        point.x >= 0 && point.y >= 0 && point.x < self.dim.x && point.y < self.dim.y
    }

    pub fn inner(&self) -> &[T] {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }

    pub fn get(&self, p: Vec2<GridNum>) -> &T {
        &self.inner[(p.x * self.dim.y + p.y) as usize]
    }
    pub fn get_mut(&mut self, p: Vec2<GridNum>) -> &mut T {
        &mut self.inner[(p.x * self.dim.y + p.y) as usize]
    }

    pub fn len(&self) -> usize {
        (self.dim.x * self.dim.y) as usize
    }
}

///A parsed map that is no longer represented as a string.
pub struct Grid2D {
    dim: Vec2<GridNum>,
    inner: BitVec,
}

impl Grid2D {
    pub fn from_str(map: Map) -> Grid2D {
        let mut grid = Grid2D::new(map.dim);

        for (y, line) in map.str.lines().enumerate() {
            for (x, c) in line.chars().enumerate() {
                match c {
                    '█' => {
                        assert!(x < map.dim.x as usize, "x too big {}");
                        assert!(y < map.dim.y as usize, "y too big {:?}", (x, y, map.dim));
                        grid.set(vec2(x, y).inner_as(), true);
                    }
                    ' ' => {}
                    _ => {
                        panic!("unknown char {:?}", c);
                    }
                }
            }
        }
        grid
    }
    pub fn new(dim: Vec2<GridNum>) -> Grid2D {
        let inner = BitVec::from_elem((dim.x * dim.y) as usize, false);

        Grid2D { dim, inner }
    }
    pub fn dim(&self) -> Vec2<GridNum> {
        self.dim
    }

    pub fn iter(&self) -> CellIterator {
        let inner = Iterator2D::new(self.dim);
        CellIterator { grid: self, inner }
    }
    pub fn get(&self, p: Vec2<GridNum>) -> bool {
        self.inner[(p.x * self.dim.y + p.y) as usize]
    }
    pub fn get_option(&self, p: Vec2<GridNum>) -> Option<bool> {
        self.inner.get((p.x * self.dim.y + p.y) as usize)
    }
    pub fn set(&mut self, p: Vec2<GridNum>, val: bool) {
        self.inner.set((p.x * self.dim.y + p.y) as usize, val)
    }
    pub fn len(&self) -> usize {
        (self.dim.x * self.dim.y) as usize
    }

    ///Find the closest empty cell by inefficiently calculating the distance to every cell
    ///and then picking the cell with the smallest distance
    pub fn find_closest_empty(&self, start: Vec2<GridNum>) -> Option<Vec2<GridNum>> {
        let mut k: Vec<_> = Iterator2D::new(self.dim())
            .filter(|a| !self.get(*a))
            .map(|a| (a, (start - a).magnitude2()))
            .collect();

        k.sort_by(|a, b| a.1.cmp(&b.1));

        k.first().map(|a| a.0)
    }

    pub fn draw_map(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = String::new();

        let mut fa = Ok(());

        fa = fa.and(writeln!(f, ""));
        for i in 0..self.dim().y {
            for j in 0..self.dim().x {
                let cc = if self.get(vec2(j, i)) { "1 " } else { "0 " };

                res.push_str(cc);
            }
            fa = fa.and(writeln!(f, "{}", res));
            res.clear();
        }
        fa
    }
}

///The type used to represent a world position.
pub type WorldNum = f32;

use core::fmt;
impl fmt::Debug for Grid2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.draw_map(f)
    }
}

///Returns the result of the grid raycast
#[derive(Copy, Clone, Debug)]
pub enum GridRayCastResult {
    Found {
        t: WorldNum,
        cell: Vec2<GridNum>,
        dirhit: CardDir,
    },
    NotFound,
}

///A way to cast a ray until it hits a cell
pub mod raycast {
    use crate::grid::*;
    
    #[derive(Copy, Clone, Debug)]
    pub struct CollideCellEvent {
        //Cell colliding with
        pub cell: Vec2<GridNum>,

        //Direction in which we are colliding with it.
        pub dir_hit: CardDir,

        //So the user can see how long the ray is now.
        pub tval: WorldNum,
    }

    pub struct RayCaster<'a> {
        grid: &'a GridViewPort,
        ray: Ray<WorldNum>,
        dir_sign: Vec2<GridNum>,
        next_dir_sign: Vec2<GridNum>,
        current_grid: Vec2<GridNum>,
        tval: WorldNum,
    }

    impl<'a> RayCaster<'a> {
        pub fn new(grid: &'a GridViewPort, ray: Ray<WorldNum>) -> RayCaster {
            let dir_sign = vec2(
                if ray.dir.x > 0.0 { 1 } else { 0 },
                if ray.dir.y > 0.0 { 1 } else { 0 },
            );
            let next_dir_sign = vec2(
                if ray.dir.x > 0.0 { 1 } else { -1 },
                if ray.dir.y > 0.0 { 1 } else { -1 },
            );

            let mut current_grid = grid.to_grid(ray.point);

            //Make it so that if the bot is on a line,
            //it will also still consider the line first
            if grid.to_grid_mod(ray.point).x == 0.0 {
                if dir_sign.x == 1 {
                    current_grid.x -= 1;
                }
            }

            if grid.to_grid_mod(ray.point).y == 0.0 {
                if dir_sign.y == 1 {
                    current_grid.y -= 1;
                }
            }

            assert_gt!(ray.dir.magnitude2(), 0.0);

            RayCaster {
                grid,
                ray,
                dir_sign,
                next_dir_sign,
                current_grid,
                tval: 0.0,
            }
        }
    }
    impl FusedIterator for RayCaster<'_> {}
    impl Iterator for RayCaster<'_> {
        type Item = CollideCellEvent;
        fn next(&mut self) -> Option<Self::Item> {
            let grid = &self.grid;
            let ray = &self.ray;
            let dir_sign = self.dir_sign;

            let next_grid = self.current_grid + dir_sign;

            let next_grid_pos = grid.to_world_topleft(next_grid);

            //A ray can be described as follows:
            //rx(t)=ray.dir.x*tval+ray.point.x
            //ry(t)=ray.dir.y*tval+ray.point.y
            //
            //The ray itself is all the points that satify those two equations,
            //where tval>0.
            //
            //As tval increases, so does the ray length.
            //
            //
            //We want to find out when a ray intersects
            //th next cell. A ray are intersect the cell either on
            //a x axis or in a y axis.
            //so we have two equations.
            //
            //Equation for when it hits the xaxis
            //next_grid_pos.x=ray.dir.x*tval+ray.point.x
            //
            //Equation for when it hits the yaxis
            //next_grid_pos.y=ray.dir.y*tval+ray.point.y
            //
            //In both cases, lets solve for tval.
            //The equation with the smaller tval is the one
            //the ray will hit first.
            //
            //If the tval for the x equation is smaller, the ray
            //will intersect the Y axis first.
            //

            let tvalx = (next_grid_pos.x - ray.point.x) / ray.dir.x;
            let tvaly = (next_grid_pos.y - ray.point.y) / ray.dir.y;

            let dir_hit;
            if (tvalx.is_finite() && tvalx < tvaly) || tvaly.is_infinite() || tvaly.is_nan() {
                if dir_sign.x == 1 {
                    //hit left side
                    dir_hit = CardDir::L;
                } else {
                    dir_hit = CardDir::R;
                    //hit right side
                }
                self.tval = tvalx;

                self.current_grid.x += self.next_dir_sign.x;
            } else if tvaly <= tvalx || tvalx.is_infinite() || tvalx.is_nan() {
                if dir_sign.y == 1 {
                    //hit top side
                    dir_hit = CardDir::U;
                } else {
                    //hit bottom side
                    dir_hit = CardDir::D;
                }
                self.tval = tvaly;
                self.current_grid.y += self.next_dir_sign.y;
            } else {
                unreachable!("{:?}, {:?}", (tvalx, tvaly), ray);
            }
            Some(CollideCellEvent {
                tval: self.tval,
                cell: self.current_grid,
                dir_hit,
            })
        }
    }
}

///A way to map a grid to world coordinates and vice versa
#[derive(Debug)]
pub struct GridViewPort {
    pub spacing: WorldNum,
    pub origin: Vec2<WorldNum>,
}
impl GridViewPort {
    pub fn to_world_topleft(&self, pos: Vec2<GridNum>) -> Vec2<WorldNum> {
        pos.inner_as() * self.spacing + self.origin
    }

    pub fn to_world_center(&self, pos: Vec2<GridNum>) -> Vec2<WorldNum> {
        pos.inner_as() * self.spacing + self.origin + vec2same(self.spacing) / 2.0
    }

    pub fn to_grid_mod(&self, pos: Vec2<WorldNum>) -> Vec2<WorldNum> {
        let k = self.to_grid(pos);
        let k = k.inner_as() * self.spacing;
        pos - k
    }
    pub fn to_grid(&self, pos: Vec2<WorldNum>) -> Vec2<GridNum> {
        let result = (pos - self.origin) / self.spacing;

        result.inner_as()
    }

    pub fn cell_radius(&self) -> WorldNum {
        self.spacing
    }
}

pub mod collide {
    use super::*;

    #[derive(PartialEq, Copy, Clone, Debug)]
    struct Foo {
        pub grid: Vec2<GridNum>,
        pub dir: CardDir,
        pub normal: Vec2<f32>,
        pub dis: f32,
    }

    fn foo(grid: Vec2<GridNum>, dir: CardDir, normal: Vec2<f32>, dis: f32) -> Foo {
        Foo {
            grid,
            dir,
            normal,
            dis,
        }
    }

    //Returns the normal vector, plus the amount of seperation.
    fn find_corner_offset(grid: &Grid2D, dim: &GridViewPort, point: Vec2<f32>) -> Option<Foo> {
        let grid_coord: Vec2<f32> = point.inner_into();
        let gg = dim.to_grid(grid_coord);
        if let Some(d) = grid.get_option(gg) {
            if d {
                impl Eq for Foo {}

                let corner = grid_coord;

                let grid_coord = dim.to_grid(corner);
                let topleft = dim.to_world_topleft(grid_coord);
                let bottomright = dim.to_world_topleft(grid_coord + vec2(1, 1));
                use CardDir::*;
                let arr = [
                    foo(gg, U, vec2(0.0, -1.0), point.y - topleft.y),
                    foo(gg, L, vec2(-1.0, 0.0), point.x - topleft.x),
                    foo(gg, D, vec2(0.0, 1.0), bottomright.y - point.y),
                    foo(gg, R, vec2(1.0, 0.0), bottomright.x - point.x),
                ];

                arr.iter()
                    .filter(|a| a.dis > 0.0)
                    .min_by(|a, b| a.dis.partial_cmp(&b.dis).unwrap())
                    .map(|a| *a)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn is_colliding(
        grid: &Grid2D,
        dim: &GridViewPort,
        bot: &Rect<f32>,
        radius: f32,
    ) -> [Option<(f32, CardDir, Vec2<f32>)>; 2] {
        let corners = bot.get_corners();
        let mut offsets: Vec<_> = corners
            .iter()
            .map(|&a| find_corner_offset(grid, dim, a))
            .filter(|a| a.is_some())
            .map(|a| a.unwrap())
            .filter(|a| a.dis > 0.0)
            .collect();

        for a in offsets.iter_mut() {
            a.dis += radius;
        }

        let max = offsets
            .iter()
            .filter(|a| {
                let next = a.grid + a.dir.into_vec();
                if let Some(d) = grid.get_option(next) {
                    if d {
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .min_by(|a, b| a.dis.partial_cmp(&b.dis).unwrap());

        let k = if let Some(max) = max {
            let o = offsets
                .iter()
                .filter(|a| a.dir != max.dir)
                .filter(|a| {
                    let next = a.grid + a.dir.into_vec();
                    if let Some(d) = grid.get_option(next) {
                        if d {
                            false
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                })
                .min_by(|a, b| a.dis.partial_cmp(&b.dis).unwrap());

            assert!(max.dis > 0.0);
            if let Some(o) = o {
                assert!(o.dis > 0.0);
                assert!(max.dir != o.dir);
                [
                    Some((max.dis, max.dir, max.normal)),
                    Some((o.dis, o.dir, o.normal)),
                ]
            } else {
                [Some((max.dis, max.dir, max.normal)), None]
            }
        } else {
            let max = offsets
                .iter()
                .min_by(|a, b| a.dis.partial_cmp(&b.dis).unwrap());
            if let Some(max) = max {
                [Some((max.dis, max.dir, max.normal)), None]
            } else {
                [None, None]
            }
        };

        k
    }
}
