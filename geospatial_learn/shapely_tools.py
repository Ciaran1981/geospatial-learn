"""
Pinched from the author below - these tools are very useful - credit to him.

I have altered the i/o to an ogr based one however

library with functions to edit and perform spatial analysis on vector data.
build around shapely and fiona libraries

Created on 2016-July-16
@author: Dirk Eilander (dirk.eilander@deltares.nl)

# build on library from https://github.com/ojdo/python-tools/blob/master/shapelytools.py
"""


from shapely.geometry import (box, LineString, MultiLineString, MultiPoint,
    Point, Polygon, MultiPolygon, shape)
import shapely.ops
import math
import numpy as np
#import ogr
from fiona import collection


# I/O
def read_geometries(fn, bbox=None):
    """
    reads to shapely geometries to features using fiona collection
    feature = dict('geometry': <shapely geometry>, 'properties': <dict with properties>
    """
    with collection(fn, "r") as c:
        ft_list = []
        c = c.items(bbox=bbox)
        for ft in c:
            if ft[1]['geometry'] is not None:
                ft_list.append(shape(ft[1]['geometry']))
    return ft_list


# simplify, reduce and merge  geometries methods
def prune_short_lines(lines, min_length, return_index=False):
    """Remove lines from a LineString DataFrame shorter than min_length.

    Deletes all lines from a list of LineStrings or a MultiLineString
    that have a total length of less than min_length. Vertices of touching
    lines are contracted towards the centroid of the removed line.

    Args:
        lines: list of LineStrings or a MultiLineString
        min_length: minimum length of a single LineString to be preserved

    Returns:
        the pruned list of lines
    """
    pruned_lines = [line for line in lines]  # converts MultiLineString to list
    index = []

    for i, line in enumerate(pruned_lines):
        if line.length < min_length:
            for n in neighbors(pruned_lines, line):
                contact_point = line.intersection(pruned_lines[n])
                pruned_lines[n] = bend_towards(pruned_lines[n],
                                               where=contact_point,
                                               to=line.centroid)
        else:
            index.append(i)

    lines_out = [line for i, line in enumerate(pruned_lines) if i in index]

    if return_index:
        return lines_out, index
    else:
        return lines_out


def linemerge(linestrings_or_multilinestrings, return_index=False):
    """ Merge list of LineStrings and/or MultiLineStrings.

    Given a list of LineStrings and possibly MultiLineStrings, merge all of
    them to a single MultiLineString.

    Args:
        list of LineStrings and/or MultiLineStrings

    Returns:
        a merged LineString or MultiLineString
    """
    lines = []
    for i, line in enumerate(linestrings_or_multilinestrings):
        if isinstance(line, MultiLineString):
            # line is a multilinestring, so append its components
            lines.extend(line)
        else:
            # line is a line, so simply append it
            lines.append(line)

    lines_merged = shapely.ops.linemerge(lines)

    if not return_index:
        return lines_merged
    else:
        index = index_spatial_join(lines_merged, lines, function='contains')
        return lines_merged, index


def extend_lines_min_length(lines, min_length=100, tolerance=1e-7):
    """lines with length smaller than min_length are extended along the neighboring line
    neighboring lines are found within a distance of tolerance to its endpoints
    """
    lines_out = []

    for ind, of in enumerate(lines):
        if of.length < min_length:
            other = [line for i, line in enumerate(lines) if i != ind]
            neighbor = [line for line in other if line.distance(of.boundary) < tolerance]

            # go with shortest neighbor if more than one
            if not len(neighbor) == 0:
                nb = neighbor[0]
                res_len = min_length - of.length

                # construct new lines
                if nb.distance(of.boundary[0]) < tolerance:
                    # cut line from end
                    nb_short, _ = cut_line(LineString(reversed(list(nb.coords))), res_len)
                    of = LineString(list(nb_short.coords) + list(of.coords)[1:])
                elif nb.distance(of.boundary[1]) < tolerance:
                    # cut line
                    nb_short, _ = cut_line(nb, res_len)
                    of = LineString(list(of.coords)[:-1] + list(nb_short.coords))
        lines_out.append(of)
    return lines_out


def polygon_union(polygons, return_index=False):
    """returns simplified union of polygons without holes"""
    # remove 'holes' in polygon
    polygons_out = []
    for ob in polygons:
        if isinstance(ob, MultiPolygon):
            for g in ob.geoms:
                pol = Polygon(list(Polygon(g).exterior.coords))
                polygons_out.append(pol)
        elif isinstance(ob, Polygon):
            pol = Polygon(list(ob.exterior.coords))
            polygons_out.append(pol)

    # merge polygons
    polygons_union = shapely.ops.cascaded_union(polygons_out)
    # return polygons_union

    if not return_index:
        return polygons_union
    else:
        index = index_spatial_join(polygons_union, polygons, function='contains')
        return polygons_union, index


def one_linestring_per_intersection(lines, return_index=False):
    """ Move line endpoints to intersections of line segments.

    Given a list of touching or possibly intersecting LineStrings, return a
    list LineStrings that have their endpoints at all crossings and
    intersecting points and ONLY there.

    Args:
        a list of LineStrings or a MultiLineString

    Returns:
        a list of LineStrings
    """
    lines_merged = shapely.ops.linemerge(lines)

    # intersecting multiline with its bounding box somehow triggers a first
    bounding_box = box(*lines_merged.bounds)

    # perform linemerge (one linestring between each crossing only)
    # if this fails, write function to perform this on a bbox-grid and then
    # merge the result
    lines_merged = lines_merged.intersection(bounding_box)
    lines_merged = shapely.ops.linemerge(lines_merged)

    if not return_index:
        return lines_merged
    else:
        index = index_spatial_join(lines_merged, lines, function='contains')
        return lines_merged, index


def reduce_line2singlevertex(lines, tolerance=None, return_index=False):
    """simplifies  a line to a single vertex
    returns a Tuple with the simplified line and the max error made
    if a value for tolerance is set, lines with an error larger than tolerance are removed"""
    lines_out = [LineString((line.coords[0], line.coords[-1])) for line in lines]
    max_dist = [max([Point(p).distance(l2) for p in l1.coords]) for l1, l2 in zip(lines, lines_out)]

    if tolerance is not None:
        check = [d <= tolerance for d in max_dist]
        lines_out = [l for l, c in zip(lines_out, check) if c]
        max_dist = [l for l, c in zip(max_dist, check) if c]

    if not return_index:
        return lines_out, max_dist
    else:
        index = [i for i, d in enumerate(max_dist) if d <= tolerance]
        return lines_out, max_dist, index


# neighborhood methods
def nearest_neighbor_within(others, point, max_distance):
    """Find nearest point among others up to a maximum distance.

    Args:
        others: a list of Points or a MultiPoint
        point: a Point
        max_distance: maximum distance to search for the nearest neighbor

    Returns:
        A shapely Point if one is within max_distance, None otherwise
    """
    search_region = point.buffer(max_distance)
    interesting_points = search_region.intersection(MultiPoint(others))

    if not interesting_points:
        closest_point = None
    elif isinstance(interesting_points, Point):
        closest_point = interesting_points
    else:
        distances = [point.distance(ip) for ip in interesting_points
                     if point.distance(ip) > 0]
        closest_point = interesting_points[distances.index(min(distances))]

    return closest_point


def closest_object(geometries, point):
    """Find the nearest geometry among a list, measured from fixed point.

    Args:
        geometries: a list of shapely geometry objects
        point: a shapely Point

    Returns:
        Tuple (geom, min_dist, min_index) of the geometry with minimum distance
        to point, its distance min_dist and the list index of geom, so that
        geom = geometries[min_index].
    """
    min_dist, min_index = min((point.distance(geom), k)
                              for (k, geom) in enumerate(geometries))

    return geometries[min_index], min_dist, min_index


# snapping methods
def snap_endings(lines, max_dist):
    """Snap endpoints of lines together if they are at most max_dist apart.

    Args:
        lines: a list of LineStrings or a MultiLineString
        max_dist: maximum distance two endpoints may be joined together
    """

    # initialize snapped lines with list of original lines
    # snapping points is a MultiPoint object of all vertices
    snapped_lines = [line for line in lines]
    snapping_points = vertices_from_lines(snapped_lines)

    # isolated endpoints are going to snap to the closest vertex
    isolated_endpoints = find_isolated_endpoints(snapped_lines)

    # only move isolated endpoints, one by one
    for endpoint in isolated_endpoints:
        # find all vertices within a radius of max_distance as possible
        target = nearest_neighbor_within(snapping_points, endpoint,
                                         max_dist)

        # do nothing if no target point to snap to is found
        if not target:
            continue

        # find the LineString to modify within snapped_lines and update it
        for i, snapped_line in enumerate(snapped_lines):
            if endpoint.touches(snapped_line):
                snapped_lines[i] = bend_towards(snapped_line, where=endpoint,
                                                to=target)
                break

        # also update the corresponding snapping_points
        for i, snapping_point in enumerate(snapping_points):
            if endpoint.equals(snapping_point):
                snapping_points[i] = target
                break

    # post-processing: remove any resulting lines of length 0
    snapped_lines = [s for s in snapped_lines if s.length > 0]

    return snapped_lines


def snap_lines(lines, max_dist, tolerance=1e-7, return_index=False):
    """Snap lines together if the endpoint of one line is at most max_dist apart from the another line.
    The line to which the endpoint is snapped will be split at the intersection

    Args:
        lines: a list of LineStrings or a MultiLineString
        max_dist: maximum distance two endpoints may be joined together
    """
    lines_out = []
    split_points = []

    for i1, line in enumerate(lines):
        other = [x for i, x in enumerate(lines) if i1 != i]
        start, end = Point(line.coords[0]), Point(line.coords[-1])

        # find closest geometry to start point
        nn_geom, dist, _ = closest_object(other, start)

        # calculate intersection if within max_distance
        if dist <= max_dist:
            new_pnt = project_point_to_object(start, nn_geom, max_dist=max_dist)
            # if intersection not at line end-points split line
            if nn_geom.boundary.distance(new_pnt) > tolerance:
                split_points.append(new_pnt)
        # add node to vertex if not connected and within max distance
        if (dist > 0) & (dist <= max_dist):
            line = LineString([(new_pnt.x, new_pnt.y)] + list(line.coords))  # add node to start

        # find closest geometry tp end point
        nn_geom, dist, _ = closest_object(other, end)
        # calculate intersection if within max_distance
        if dist <= max_dist:
            new_pnt = project_point_to_object(end, nn_geom, max_dist=max_dist)
            # if intersection not at line end-points split line
            if nn_geom.boundary.distance(new_pnt) > tolerance:
                split_points.append(new_pnt)
        # add node to vertex if not connected and within max distance
        if (dist > 0) & (dist <= max_dist):
            line = LineString(list(line.coords) + [(new_pnt.x, new_pnt.y)])  # add node to end

        lines_out.append(line)

    # post-process split lines if no node at intersection
    if not return_index:
        return remove_redundant_nodes(split_lines_points(lines_out, split_points))
    else:
        lines, index = split_lines_points(lines_out, split_points, return_index=return_index)
        lines = remove_redundant_nodes(lines, tolerance)
    return lines, index


def snap_points2geometries(points, geometries, max_dist, return_index=False):
    """"""
    out_list = []
    index = []
    # input list of point geometries
    if isinstance(points[0], Point):
        for i, p in enumerate(points):
            line_snap = closest_object(geometries, p)[0]  # find closest objects and snap to line
            pnt_snap = project_point_to_object(p, line_snap, max_dist=max_dist)
            if pnt_snap is not None:
                out_list.append(pnt_snap)
                index.append(i)
    if not return_index:
        return out_list
    else:
        return out_list, index


def project_point_to_line(point, line_start, line_end):
    """Find nearest point on a straight line, measured from given point.

    Args:
        point: a shapely Point object
        line_start: the line starting point as a shapely Point
        line_end: the line end point as a shapely Point

    Returns:
        a shapely Point that lies on the straight line closest to point

    Source: http://gis.stackexchange.com/a/438/19627
    """
    line_magnitude = line_start.distance(line_end)

    u = ((point.x - line_start.x) * (line_end.x - line_start.x) +
         (point.y - line_start.y) * (line_end.y - line_start.y)) \
         / (line_magnitude ** 2)

    # closest point does not fall within the line segment,
    # take the shorter distance to an endpoint
    if u < 0.00001 or u > 1:
        ix = point.distance(line_start)
        iy = point.distance(line_end)
        if ix > iy:
            return line_end
        else:
            return line_start
    else:
        ix = line_start.x + u * (line_end.x - line_start.x)
        iy = line_start.y + u * (line_end.y - line_start.y)
        return Point([ix, iy])


def project_point_to_object(point, geometry, max_dist=float("inf")):
    """Find nearest point in geometry, measured from given point.

    Args:
        point: a shapely Point
        geometry: a shapely geometry object (LineString, Polygon)

    Returns:
        a shapely Point that lies on geometry closest to point
    """
    nearest_point = None
    min_dist = float("inf")

    if isinstance(geometry, Polygon):
        for seg_start, seg_end in pairs(list(geometry.exterior.coords)):
            line_start = Point(seg_start)
            line_end = Point(seg_end)

            intersection_point = project_point_to_line(point, line_start, line_end)
            cur_dist = point.distance(intersection_point)

            if (cur_dist < min_dist) & (cur_dist < max_dist):
                min_dist = cur_dist
                nearest_point = intersection_point

    elif isinstance(geometry, LineString):
        for seg_start, seg_end in pairs(list(geometry.coords)):
            line_start = Point(seg_start)
            line_end = Point(seg_end)

            intersection_point = project_point_to_line(point, line_start, line_end)
            cur_dist = point.distance(intersection_point)

            if (cur_dist < min_dist) & (cur_dist < max_dist):
                min_dist = cur_dist
                nearest_point = intersection_point
    else:
        raise NotImplementedError("project_point_to_object not implemented for"+
                                  " geometry type '" + geometry.type + "'.")

    return nearest_point


def bend_towards(line, where, to):
    """Move the point where along a line to the point at location to.

    Args:
        line: a LineString
        where: a point ON the line (not necessarily a vertex)
        to: a point NOT on the line where the nearest vertex will be moved to

    Returns:
        the modified (bent) line
    """

    if not line.contains(where) and not line.touches(where):
        raise ValueError('line does not contain the point where.')

    coords = line.coords[:]
    # easy case: where is (within numeric precision) a vertex of line
    for k, vertex in enumerate(coords):
        if where.almost_equals(Point(vertex)):
            # move coordinates of the vertex to destination
            coords[k] = to.coords[0]
            return LineString(coords)

    # hard case: where lies between vertices of line, so
    # find nearest vertex and move that one to point to
    _, min_k = min((where.distance(Point(vertex)), k)
                           for k, vertex in enumerate(coords))
    coords[min_k] = to.coords[0]
    return LineString(coords)


# split geometries methods
def split_lines_point(lines, point, tolerance=1e-3, return_index=False):
    """Split line at a given point

    Args:
        lines: a list of shapely LineStrings or a MultiLineString
        point: shapely Point
        tolerance: required to check if segment intersects with line

    Returns:
        a list of LineStrings
    """
    # find line which intersects with point, but not with start of end point
    idx = [i for i, line in enumerate(lines) if
           (line.distance(point) <= tolerance) & (line.boundary.distance(point) > tolerance)]

    if len(idx) == 0:
        raise Warning('line does not intersect with point with given tolerance.')
    elif len(idx) > 1:
        raise Warning('more than one line within given tolerance')
    else:
        idx = idx[0]
        lines_out = [line for i, line in enumerate(lines) if i != idx]
        index = [i for i, line in enumerate(lines) if i != idx] + [idx, idx]

    # for intersecting line, find intersecting segment and split line
    coords = list(lines[idx].coords)
    segments = [LineString(s) for s in pairs(coords)]
    n = len(segments)
    for i, segment in enumerate(segments):
        # find intersecting segment
        if segment.distance(point) <= tolerance:
            # split line at vertex if within tolerance
            if Point(coords[i]).distance(point) <= tolerance:
                lines_out.append(LineString(coords[:i+1]))
                lines_out.append(LineString(coords[i:]))
                break
            # split line at point
            else:
                lines_out.append(LineString(coords[:i+1] + [(point.x, point.y)]))
                lines_out.append(LineString([(point.x, point.y)] + coords[i+1:]))
                break

    if not return_index:
        return lines_out
    else:
        return lines_out, index


def add_node(lines, point, tolerance=1e-3):
    """Split line at a given point

    Args:
        lines: a list of shapely LineStrings or a MultiLineString
        point: shapely Point
        tolerance: required to check if segment intersects with line

    Returns:
        a list of LineStrings
    """
    # find line which intersects with point, but not with start of end point
    idx = [i for i, line in enumerate(lines) if
           (line.distance(point) <= tolerance) & (line.boundary.distance(point) > tolerance)]

    if len(idx) == 0:
        raise Warning('line does not intersect with point with given tolerance.')
    elif len(idx) > 1:
        raise Warning('more than one line within given tolerance')
    else:
        idx = idx[0]

    # for intersecting line, find intersecting segment and split line
    coords = list(lines[idx].coords)
    segments = [LineString(s) for s in pairs(coords)]
    n = len(segments)
    for i, segment in enumerate(segments):
        # find intersecting segment
        if segment.distance(point) <= tolerance:
            # break if at point at existing node (do nothing)
            if Point(coords[i]).distance(point) <= tolerance:
                break
            # otherwise et node
            else:
                lines[idx] = LineString(coords[:i+1] + [(point.x, point.y)] + coords[i+1:])
                break

    return lines


def add_nodes(lines, point_list, tolerance=1e-3):
    """split lines at list of points, see split_lines_point for more information"""
    if isinstance(point_list, Point):
        point_list = [point_list]

    for point in point_list:
        lines = add_node(lines, point, tolerance)
    return lines


def split_lines_points(lines, point_list, tolerance=1e-3, return_index=False):
    """split lines at list of points, see split_lines_point for more information"""
    if isinstance(point_list, Point):
        point_list = [point_list]
    index = range(len(lines))

    if not return_index:
        for point in point_list:
            lines = split_lines_point(lines, point, tolerance)
        return lines
    else:
        for point in point_list:
            lines, idx0 = split_lines_point(lines, point, tolerance, return_index)
            index = [index[i] for i in idx0]
        return lines, index


def split_lines_angle(lines, split_angle=90, return_index=False):
    """Split line at vertex if angle smaller than 'split_angle'

    Args:
        lines: a list of shapely LineStrings or a MultiLineString
        split_angle: max angle between two segments

    Returns:
        a list of LineStrings
    """
    lines_out = []
    index = []

    # loop through lines
    for ind, line in enumerate(lines):
        # check if line has only single segment
        if len(list(line.coords)) == 2:
            lines_out.append(line)
            index.append(ind)
        else:
            angles, points = segment_angles(line)
            idx = np.where(np.array(angles) <= split_angle)[0]
            line = [line]
            if len(idx) >= 1:
                for i in idx:
                    line = split_lines_point(line, Point(points[i]))
            lines_out += line
            for l in line:
                index.append(ind)

    if not return_index:
        return lines_out
    else:
        return lines_out, index


def cut_line(line, distance):
    """Cuts a line in two at a distance from its starting point"""
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return LineString(coords[:i+1]), LineString(coords[i:])
        elif pd > distance:
            cp = line.interpolate(distance)
            return LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])
        elif line.length < distance:
            return line, LineString()  # return empty LineString so that output is always consistent


def explode_lines(lines, length, min_length, equal_length=False, remove_too_small=False, return_index=False):
    """explodes a LineString in to many Linestring with given length.
    If final part of line is smaller than min_length it is merged with previous

    Args:
        lines: a list of shapely LineStrings or a MultiLineString
        length: length of new lines
        min_length: minimal length of last part of a line

    Returns:
        a list of LineStrings
    """
    lines_out = []
    index = []

    if equal_length:
        for line in lines:
            n = max(1.0, round(line.length/float(length)))
            length2 = line.length/n
    else:
        length2 = length

    for i, line in enumerate(lines):
        # lines shorter than length
        if line.length <= length2:
            # shorter than min_length
            if remove_too_small and (line.length < min_length):
                continue
            # between min_length and length or remove_too_small == False
            else:
                lines_out.append(line)
                index.append(i)
        # lines longer than length
        else:
            l = line
            while (min_length+length) <= l.length:
                l1, l = cut_line(l, length2)
                lines_out.append(l1)
                index.append(i)
            # if l.length/2. >= min_length:
            #     l1, l = cut_line(l, l.length/2.)
            #     lines_out.append(l1)
            #     index.append(i)
            lines_out.append(l)
            index.append(i)


    if not return_index:
        return lines_out
    else:
        return lines_out, index


def explode_polygons(polygons, return_index=False):
    """returns main line features that make up the polygons"""
    lines_out = []
    index = []

    if isinstance(polygons, Polygon):
        polygons = [polygons]
    for i, l in enumerate(polygons):
        # get inner lines
        for p in l.interiors:
            lines_out.append(LineString(p.coords))
            index.append(i)
        # get outer lines
        lines_out.append(LineString(l.exterior.coords))
        index.append(i)

    if not return_index:
        return lines_out
    else:
        return lines_out, index


# clip geometries methods
def clip_lines_with_polygon(lines, polygon, tolerance=1e-3, within=True, return_index=False):
    """clip lines based on polygon outline"""
    # Multipolygon to list of polygons
    # if isinstance(polygon, MultiPolygon):
    #     raise ValueError("function works with tyype polygon only")
    # get boundaries of polygon
    boundaries = explode_polygons(polygon)
    # find intersection points of boundaries and lines and split lines based on it
    intersection_points = line_intersections(boundaries, lines)
    lines_split, index = split_lines_points(lines, intersection_points, tolerance=tolerance, return_index=True)
    # select lines that are contained by polygon
    polygon_buffer = polygon.buffer(0.05)  # small buffer to allow for use 'within' function
    if within:
        lines_clip = [line for line in lines_split if line.within(polygon_buffer)]
        index = [i for i, line in zip(index, lines_split) if line.within(polygon_buffer)]
    else:
        lines_clip = [line for line in lines_split if not line.within(polygon_buffer)]
        index = [i for i, line in zip(index, lines_split) if not line.within(polygon_buffer)]
    if not return_index:
        return lines_clip
    else:
        return lines_clip, index


def offset_and_clip(lines, offset_dist=30, buffer_dist=29.9, side='both', join_style=2):
    """offset lines, but clip where these lines are within buffer distance of other lines"""
    # calc buffer around lines
    cbuffer = polygon_union([c.buffer(buffer_dist, join_style=join_style) for c in lines])

    if side == 'both':
        sides = ['left', 'right']
    else:
        sides = [side]

    lines_out = []
    for s in sides:
        # offset canal lines
        offset_lines = [c.parallel_offset(offset_dist, s, join_style=join_style) for c in lines]
        #clip lines outside canal buffer
        lines_out.append(clip_lines_with_polygon(offset_lines, cbuffer, within=False))

    if side == 'both':
        return lines_out[0], lines_out[1]
    else:
        return lines_out[0]


# utils
def perpendicular_line(l1, length):
    """Create a new Line perpendicular to this linear entity which passes
    through the point `p`.


    """
    dx = l1.coords[1][0] - l1.coords[0][0]
    dy = l1.coords[1][1] - l1.coords[0][1]

    p = Point(l1.coords[0][0] + 0.5*dx, l1.coords[0][1] + 0.5*dy)
    x, y = p.coords[0][0],  p.coords[0][1]

    if (dy == 0) or (dx == 0):
        a = length / l1.length
        l2 = LineString([(x - 0.5*a*dy, y - 0.5*a*dx),
                         (x + 0.5*a*dy, y + 0.5*a*dx)])

    else:
        s = -dx/dy
        a = ((length * 0.5)**2 / (1 + s**2))**0.5
        l2 = LineString([(x + a, y + s*a),
                         (x - a, y - s*a)])

    return l2


def pairs(lst):
    """Iterate over a list in overlapping pairs.

    Args:
        lst: an iterable/list

    Returns:
        Yields a pair of consecutive elements (lst[k], lst[k+1]) of lst. Last
        call yields (lst[-2], lst[-1]).

    Example:
        lst = [4, 7, 11, 2]
        pairs(lst) yields (4, 7), (7, 11), (11, 2)

    Source:
        http://stackoverflow.com/questions/1257413/1257446#1257446
    """
    i = iter(lst)
    prev = next(i)
    for item in i:
        yield prev, item
        prev = item


def neighbors(lines, of):
    """Find the indices in a list of LineStrings that touch a given LineString.

    Args:
        lines: list of LineStrings in which to search for neighbors
        of: the LineString which must be touched

    Returns:
        list of indices, so that all lines[indices] touch the LineString of
    """
    return [k for k, line in enumerate(lines) if line.touches(of)]


def endpoints_from_lines(lines):
    """Return list of terminal points from list of LineStrings."""

    all_points = []
    for line in lines:
        for i in [0, -1]: # start and end point
            all_points.append(line.coords[i])

    unique_points = set(all_points)

    return [Point(p) for p in unique_points]


def vertices_from_lines(lines):
    """Return list of unique vertices from list of LineStrings."""

    vertices = []
    for line in lines:
        vertices.extend(list(line.coords))
    return [Point(p) for p in set(vertices)]


def find_isolated_endpoints(lines):
    """Find endpoints of lines that don't touch another line.

    Args:
        lines: a list of LineStrings or a MultiLineString

    Returns:
        A list of line end Points that don't touch any other line of lines
    """

    isolated_endpoints = []
    for i, line in enumerate(lines):
        other_lines = lines[:i] + lines[i+1:]
        for q in [0, -1]:
            endpoint = Point(line.coords[q])
            if any(endpoint.touches(another_line)
                   for another_line in other_lines):
                continue
            else:
                isolated_endpoints.append(endpoint)
    return isolated_endpoints


def line_intersections(lines1, lines2=None):
    """ creates list with points of line intersections

    :param lines1: MultiLineString or list of lines
    :param lines2: MultiLineString or list of lines, if None find intersections amongst lines1
    :return:        list with shapely points of intersection
    """
    points = []
    for i1, l1 in enumerate(lines1):
        if lines2 is None:
            lines3 = [x for i, x in enumerate(lines1) if i != i1]
        else:
            lines3 = lines2

        for l3 in lines3:
            pnt = l1.intersection(l3)
            if not pnt.is_empty:
                if isinstance(pnt, MultiPoint):
                    for g in pnt.geoms:
                        points.append(Point(g))
                else:
                    points.append(pnt)
    return points


def segment_angles(line):
    """calculate angles between all segments of a line

    :param line:    a shapely LineStrings
    :return:        a list angles and points
    """
    # make pairs of segment coordinates
    segmentpairs = list(pairs([s for s in pairs(line.coords)]))
    # find coordinates of vertices between two segments
    points = [np.array(cs[0][1]) for cs in segmentpairs]
    # calculate vectors by transformation of middle point to (0,0) coordinates
    vectors = [(np.array(cs[0][0])-p0, np.array(cs[1][1])-p0)
               for cs, p0 in zip(segmentpairs, points)]
    # calculate angle between vectors
    angles = [angle(v[0], v[1]) for v in vectors]
    return angles, points


def remove_redundant_nodes(lines, tolerance=1e-7):
    """remove vertices with length smaller than tolerance"""
    lines_out = []
    for line in lines:
        coords = line.coords
        l_segments = np.array([Point(s[0]).distance(Point(s[1])) for s in pairs(coords)])
        idx = np.where(l_segments < tolerance)[0]
        lines_out.append(LineString([c for i, c in enumerate(coords) if i not in idx]))
    return lines_out


def angle(v1, v2):
    """return angle between vector v1 and vector v2"""
    return math.atan2(np.abs(np.cross(v1, v2)), np.dot(v1, v2))/math.pi*180


def index_spatial_join(geometries, geometries_sample, function='within'):
    """find index to map attributes from geometries_sample to geometries based on  geospatial relation

    :param geometries:         geometries to be updated
    :param geometries_sample:  feature list property values from
    :param function:        string with name of shapely binary predicates to define spatial relation,
                            options are: 'within', 'contains', 'crosses', 'disjoint', 'equals', 'intersects', 'touches'
                            see: http://toblerity.org/shapely/manual.html#binary-predicates
    :return:                updated feature list
    """
    index = []
    func_list = ['within', 'contains', 'crosses', 'disjoint', 'equals', 'intersects', 'touches']
    if isinstance(geometries, (Point, LineString, Polygon)):
        geometries = [geometries]

    if function in func_list:
        # set function to define spatial relation
        spatial_relate = eval('lambda x, y: x.'+function+'(y)')
        for geom in geometries:
            # find feature in sample list that matches spatial relation
            index.append(np.where([spatial_relate(geom, geom_sample) for geom_sample in geometries_sample])[0].tolist())

    elif function is 'NN':
        for geom in geometries:
            # TODO: check if this also works for other than point objects
            # TODO: build in max distance check
            index.append(closest_object(geometries_sample, geom)[2])  # find nearest object from list
    else:
        raise ValueError("unknown spatial relation function, choose from: {:s}".format(', '.join(func_list)))

    return index
