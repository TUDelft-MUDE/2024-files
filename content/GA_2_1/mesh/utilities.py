import numpy as np
import timeit
import matplotlib.pyplot as plt


class Mesh:
    def __init__(self, coordinates, side_length, boundaries):
        self.coordinates = coordinates
        self.side_length = side_length
        self.triangles = None
        self.shared_sides = None
        self.centroids = None
        self.side_midpoints = None
        self.refinement_number = 0
        self.check_initialization()
        self.get_boundary_coordinates(boundaries)
        self.define_triangles()
        self.define_shared_sides()
        self.get_all_sides()
        self.get_boundary_sides()
        self.set_initial_conditions()

    def solve(self, t_final, Nt, D):
        unknowns = np.zeros((Nt+1, len(self.triangles)))
        unknowns[0, :] = self.initial_conditions

        dt = t_final/Nt

        for time_step in range(Nt):
            for triangle_id, triangle in enumerate(self.triangles):
                phi = unknowns[time_step, triangle_id]
                flux = np.zeros(3)
                for i in range(3):
                    side = [triangle[i], triangle[(i+1)%3]]
                    
                    side_id = None
                    for idx, shared_side in enumerate(self.shared_sides):
                        if sorted(side) == sorted(shared_side[0]):
                            side_id = idx
                            other_triangle_id = self.shared_sides[side_id][1][0] if self.shared_sides[side_id][1][1] == triangle_id else self.shared_sides[side_id][1][1]
                            break

                    if side_id is not None:
                        phi_neighbor = unknowns[time_step, other_triangle_id]
                        area = self.get_triangle_area(triangle_id, self.triangles)
                        side_length = np.linalg.norm(self.coordinates[side[0]] - self.coordinates[side[1]])
                        centroid_distance = np.linalg.norm(self.centroids[other_triangle_id] - self.centroids[triangle_id])
                        flux[i] = D*(phi_neighbor - phi)*side_length/centroid_distance/area

                    else:
                        for idx, boundary_side in enumerate(self.boundary_sides):
                            if sorted(side) == sorted(self.all_sides[boundary_side]):
                                if self.boundary_side_types[idx][0] == 'Neumann':
                                    flux[i] = self.boundary_side_types[idx][1]
                                else:
                                    print(f'WARNING: triangle {triangle_id}, boundary side {idx}, side ({side[0]}, {side[1]}) not found in any side libraries!')
                unknowns[time_step+1, triangle_id] = phi + dt*np.sum(flux)
        self.unknowns = unknowns
        self.Nt = Nt
        self.t_final = t_final
        print('Solving complete!')
        print(f'  t_final = {t_final}, Nt = {Nt}, D = {D}')
        return unknowns
                    
                

    def try_triangles(self, triangles=None, triangle_id=None):
        if triangles is None:
            assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Define with method define_triangles(), or \n'
                +'--> Provide as keyword argument triangles=[[],[],...]\n')
            triangles = self.triangles
        if triangle_id is None:
            triangle_id = range(len(triangles))
        self.plot_triangles(triangles=triangles, triangle_id=triangle_id)
        self.check_triangles(triangles=triangles, triangle_id=triangle_id)

    def define_triangles(self):
        triangles = []
        tol = self.side_length/20
        for i, a in enumerate(self.coordinates):
            for j, b in enumerate(self.coordinates):
                if i != j and np.isclose(np.linalg.norm(a - b), self.side_length, atol=tol):
                    for k, c in enumerate(self.coordinates):
                        if k != i and k != j and np.isclose(np.linalg.norm(a - c), self.side_length, atol=tol) and np.isclose(np.linalg.norm(b - c), self.side_length, atol=tol):
                            if sorted([i, j, k]) in triangles:
                                continue
                            triangles.append(sorted([i, j, k]))
        self.triangles = triangles
        self.unknowns = np.zeros(len(triangles))
        self.check_triangles(triangles=triangles, report=False)
        return triangles
    
    def get_boundary_coordinates(self, boundaries):
        boundary_coordinates = []
        boundary_types = []
        for boundary in boundaries:
            boundary_types.append(boundary[0])
            coordinates_i = np.zeros((len(boundary[1]), 2))
            for i in range(len(boundary[1])):
                coordinates_i[i] = self.coordinates[boundary[1][i]]
            boundary_coordinates.append(coordinates_i)
        self.boundaries = boundary_coordinates
        self.boundary_types = boundary_types
        return boundary_coordinates

    def get_triangle_area(self, triangle_id, triangles):
        triangle = triangles[triangle_id]
        a = self.coordinates[triangle[0]]
        b = self.coordinates[triangle[1]]
        c = self.coordinates[triangle[2]]
        return 0.5*np.abs(np.cross(b-a, c-a))
    
    def get_centroids(self):
        assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Define with method define_triangles(), or \n'
                +'--> Provide as keyword argument triangles=[[],[],...]\n')
        self.centroids = np.zeros((len(self.triangles), 2))
        for i, triangle in enumerate(self.triangles):
            a = self.coordinates[triangle[0]]
            b = self.coordinates[triangle[1]]
            c = self.coordinates[triangle[2]]
            centroid = (a + b + c) / 3
            self.centroids[i] = centroid
        return self.centroids
    
    def get_side_midpoints(self):
        assert self.shared_sides is not None, (
                'NO SIDES DEFINED!\n'
                +'--> Define with method define_shared_sides(), or \n'
                +'--> Provide as keyword argument sides=[[[],[]],...]\n')
        self.side_midpoints = np.zeros((len(self.shared_sides), 2))
        for i, side in enumerate(self.shared_sides):
            triangle_1_id, triangle_2_id = side[1]
            bar_1 = self.get_centroids()[triangle_1_id]
            bar_2 = self.get_centroids()[triangle_2_id]
            midpoint = (bar_1 + bar_2) / 2
            self.side_midpoints[i] = midpoint
        return self.side_midpoints
    
    def set_initial_conditions(self, default=0, special_triangles=None):
        """Set initial conditions.
        
        Can only set conditions based on triangles in initial mesh.
        Cannot subdivide by sub-triangles after mesh is refined.
        In other words, if you refine the mesh, all 4 sub-triangles
        will have same initial condition.
        
        By default all triangles will be set to value default=0.
        If special_triangles is provided, these triangles will be
        set to value specified in the list:
            [[triangle_index, initial_value], [...], ...].
        """
        assert self.triangles is not None, (
            'NO TRIANGLES DEFINED!\n'
            +'--> Define with method define_triangles(), or \n'
            +'--> Provide as keyword argument triangles=[[],[],...]\n')
        triangles = self.triangles

        initial_conditions = np.zeros(len(triangles))
        if default != 0:
            initial_conditions[:] = default
        if special_triangles is not None:
            for triangle in special_triangles:
                initial_conditions[triangle[0]] = triangle[1]

        self.original_triangles = triangles
        self.original_coordinates = self.coordinates
        self.original_initial_conditions = initial_conditions
        self.get_initial_conditions()
        return initial_conditions
    
    def get_initial_conditions(self):
        """Initial conditions for current set of triangles.
        
        get the centroids, then find the original_triangle each
        centroid is within and set the value to the intital
        condition for the original_triangle
        
        """
        assert self.original_initial_conditions is not None, (
            'NO INITIAL CONDITIONS SET!\n'
            +'--> Set with method set_initial_conditions(), or \n'
            )
        self.get_centroids()
        initial_conditions = np.zeros(len(self.triangles))
        for i, centroid in enumerate(self.centroids):
            original_triangle_id = self.centroid_in_triangle(centroid)
            initial_conditions[i] = self.original_initial_conditions[original_triangle_id]
        self.initial_conditions = initial_conditions
        return initial_conditions

    def centroid_in_triangle(self, centroid):
        """return the triangle index for a given centroid
        
        centroid is a list or array of length 2
        uses original triangle geometry:
          - original_tirangles
          - original_coordinates
          - original_initial_conditions
        
        """

        for i, triangle in enumerate(self.original_triangles):
            a = self.original_coordinates[triangle[0]]
            b = self.original_coordinates[triangle[1]]
            c = self.original_coordinates[triangle[2]]
            if self.point_in_triangle(centroid, a, b, c):
                return i
        return None
    
    def point_in_triangle(self, p, a, b, c):
        def sign(p, p1, p2):
            return (p1[0] - p[0]) * (p2[1] - p[1]) - (p2[0] - p[0]) * (p1[1] - p[1])
        b1 = sign(p, a, b) < 0.0
        b2 = sign(p, b, c) < 0.0
        b3 = sign(p, c, a) < 0.0
        return ((b1 == b2) and (b2 == b3))

    def plot_everything(self):
        fig = self.plot_coordinates()
        ax = fig.get_axes()[0]
        if self.centroids is not None:
            ax.scatter(self.centroids[:, 0],
                       self.centroids[:, 1],
                       label='Bars',
                       marker='s',
                       color='red')
        if self.side_midpoints is not None:
            ax.scatter(self.side_midpoints[:, 0],
                       self.side_midpoints[:, 1],
                       label='Side Midpoints',
                       marker='^',
                       color='orange')
        ax.legend()
        ax.set_title('My Plan for the City')
        return fig

    def plot_coordinates(self, show_labels=True):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.coordinates[:, 0],
                   self.coordinates[:, 1],
                   label='Student Houses',
                   color='blue')
        ax.set_title('Triangle Vertices')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        text_offset = 0.1
        if show_labels:
            for i in range(self.coordinates.shape[0]):
                ax.text(self.coordinates[i, 0] - text_offset,
                        self.coordinates[i, 1] - text_offset,
                        str(i),
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=12, color='black')
        
        ax.grid(True)
        ax.axis('equal')
        # ax.set_xlim(0, 8)
        # ax.set_ylim(0, 6)
        return fig
    
    def plot_boundaries(self):
        fig = self.plot_coordinates()
        ax = fig.get_axes()[0]
        for boundary in self.boundaries:
            ax.plot(boundary[:, 0], boundary[:, 1], color='black', linestyle='-', linewidth=2)
        ax.set_title('Boundaries')
        return fig
    
    def plot_boundary_sides(self):
        fig = self.plot_coordinates()
        ax = fig.get_axes()[0]
        for side in self.boundary_sides:
            start = self.coordinates[self.all_sides[side][0]]
            end = self.coordinates[self.all_sides[side][1]]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                color='black', linestyle='-', linewidth=2)
        ax.set_title('Boundary Sides')
        return fig
    
    def plot_triangles(self, triangles=None, triangle_id=None,
                       fill_color=False, time_step=-1,
                       show_labels=True):
        if triangles is None:
            assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Define with method define_triangles(), or \n'
                +'--> Provide as keyword argument triangles=[[],[],...]\n'
        )
            triangles = self.triangles
        if triangle_id is None:
            triangle_id = range(len(triangles))
        fig = self.plot_coordinates(show_labels=show_labels)
        ax = fig.get_axes()[0]
        if isinstance(triangle_id, int):
            ax.set_title(f'Triangle {triangle_id} (vertices {triangles[triangle_id]})')
            triangle_id = [triangle_id]
        else:
            ax.set_title(f'Triangles plotted: {triangle_id}')
        
        colors = []
        if fill_color == 'initial_conditions':
            min_val = np.min(self.initial_conditions)
            max_val = np.max(self.initial_conditions)
            norm = plt.Normalize(min_val, max_val)
            cmap = plt.get_cmap('viridis')
            colors = [cmap(norm(val)) for val in self.initial_conditions]
            ax.set_title('Initial Conditions')
        elif fill_color == 'unknowns':
            min_val = np.min(self.original_initial_conditions)
            max_val = np.max(self.original_initial_conditions)
            if min_val > np.min(self.unknowns[time_step]):
                min_val = np.min(self.unknowns[time_step])
                print('NOTE: min value color scale adjusted below min initial value')
            if max_val < np.min(self.unknowns[time_step]):
                max_val = np.max(self.unknowns[time_step])
                print('NOTE: min value color scale adjusted below min initial value')
            if time_step < 0:
                print_time = self.Nt + time_step
            else:
                print_time = time_step/self.Nt*self.t_final
            
            ax.set_title(f'Solution at time t = {print_time:.3f} '
                         +f'(index {time_step}, max time {self.t_final})')
                

            norm = plt.Normalize(min_val, max_val)
            cmap = plt.get_cmap('inferno')
            colors = [cmap(norm(val)) for val in self.unknowns[time_step]]
        else:
            colors = 'green'

        for t_id in triangle_id:
            triangle = triangles[t_id]
            for i in range(3):
                start = self.coordinates[triangle[i]]
                end = self.coordinates[triangle[(i+1)%3]]
                ax.plot([start[0], end[0]], [start[1], end[1]],
                    color='black', linestyle='--', linewidth=2)
            if fill_color:
                ax.fill(self.coordinates[triangle, 0], 
                    self.coordinates[triangle, 1], 
                    color=colors[t_id], alpha=0.3)
            else:
                ax.fill(self.coordinates[triangle, 0], 
                    self.coordinates[triangle, 1], 
                    color='green', alpha=0.3)
        if fill_color:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label('Temperature')
            
        return fig
         
    def define_shared_sides(self):
        assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Define with method define_triangles(), or \n'
        )
        shared_sides = []
        tol = self.side_length/20
        for i, tri1 in enumerate(self.triangles):
            for j, tri2 in enumerate(self.triangles):
                if i >= j:
                    continue
                common_vertices = list(set(tri1).intersection(set(tri2)))
                if len(common_vertices) == 2:
                    shared_sides.append([common_vertices, [i, j]])

        self.shared_sides = shared_sides
        self.check_shared_sides()
        return shared_sides
    
    def get_all_sides(self):
        assert self.triangles is not None, (
            'NO TRIANGLES DEFINED!\n'
            +'--> Define with method define_triangles(), or \n'
            +'--> Provide as keyword argument triangles=[[],[],...]\n')
        sides = []
        for triangle in self.triangles:
            for i in range(3):
                side = sorted([triangle[i], triangle[(i+1)%3]])
                if side not in sides:
                    sides.append(side)
        self.all_sides = sides
        return sides
    
    def get_boundary_sides(self):
        boundary_sides = []
        boundary_types = []
        for i, side in enumerate(self.all_sides):
            on_line = False
            side_coord = [self.coordinates[side[0]]]
            side_coord.append(self.coordinates[side[1]])
            for j, boundary in enumerate(self.boundaries):
                if self.is_line_on_line(side_coord, boundary):
                    boundary_sides.append(i)
                    boundary_types.append(self.boundary_types[j])
        self.boundary_sides = boundary_sides
        self.boundary_side_types = boundary_types
        return boundary_sides
    
    def is_line_on_line(self, line1, line2):
        on_line = False
        for i in range(len(line2) - 1):
            if self.is_colinear(line1, [line2[i], line2[i+1]]):
                on_line = True
        return on_line
        
    def is_colinear(self, line1, line2):

        if np.isclose(np.cross(line1[1] - line1[0], line2[1] - line2[0]), 0):
            if np.isclose(np.cross(line2[0] - line1[0], line1[1] - line1[0]), 0):
                if line1[0][0] <= max(line2[0][0], line2[1][0]) and line1[0][0] >= min(line2[0][0], line2[1][0]):
                    if line1[0][1] <= max(line2[0][1], line2[1][1]) and line1[0][1] >= min(line2[0][1], line2[1][1]):
                        if line1[1][0] <= max(line2[0][0], line2[1][0]) and line1[1][0] >= min(line2[0][0], line2[1][0]):
                            if line1[1][1] <= max(line2[0][1], line2[1][1]) and line1[1][1] >= min(line2[0][1], line2[1][1]):
                                return True
        return False
        


    def refine_mesh(self, report=True):
        assert self.triangles is not None, (
            'NO TRIANGLES DEFINED!\n'
            +'--> Define with method define_triangles(), or \n'
            +'--> Provide as keyword argument triangles=[[],[],...]\n')
        
        if report:
            print('Refining mesh...initial status:')
            print('  refinements: ', self.refinement_number)
            print('  points:      ', self.coordinates.shape[0])
            print('  triangles:   ', len(self.triangles))
            print('  sides:       ', len(self.all_sides))
            print('  side length: ', self.side_length)


        start_time = timeit.default_timer()

        new_coordinates = np.zeros((0, 2))
        for side in self.all_sides:
            midpoint = (self.coordinates[side[0]] + self.coordinates[side[1]]) / 2
            new_coordinates = np.vstack((new_coordinates, midpoint))

        self.old_coordinates = self.coordinates

        self.coordinates = np.vstack((self.coordinates, new_coordinates))
        self.refinement_number += 1
        self.side_length /= 2

        self.define_triangles()
        self.get_all_sides()
        self.define_shared_sides()
        self.get_initial_conditions()
        self.get_boundary_sides()

        elapsed = timeit.default_timer() - start_time

        if report:
            print('Refinement complete...final status:')
            print('  refinements: ', self.refinement_number)
            print('  points:      ', self.coordinates.shape[0])
            print('  triangles:   ', len(self.triangles))
            print('  sides:       ', len(self.all_sides))
            print('  side length: ', self.side_length)
            print(f'Time taken for refinement: {elapsed:.6f} seconds')
        

        return self.coordinates

        

    def plot_shared_sides(self, sides=None, side_id=None):
        if sides is None:
            assert self.shared_sides is not None, (
                'NO SIDES DEFINED!\n'
                +'--> Define with method define_shared_sides(), or \n'
                +'--> Provide as keyword argument sides=[[[],[]],...]\n'
        )
            sides = self.shared_sides
        if side_id is None:
            side_id = range(len(sides))
        elif isinstance(side_id, int):
            side_id = [side_id]

        unique_triangles = list(set([triangle for side in sides for triangle in side[1]]))

        fig = self.plot_triangles(triangle_id=unique_triangles)
        ax = fig.get_axes()[0]

        if isinstance(side_id, int):
            ax.set_title((f'Side {side_id} (vertices {sides[side_id][0]}, '
                          +f'triangles {sides[side_id][0]})'))
        else:
            ax.set_title((f'Sides plotted: {side_id} and '
                         +f'triangles: {unique_triangles}'))

        for s_id in side_id:
            side = sides[s_id]
            start = self.coordinates[side[0][0]]
            end = self.coordinates[side[0][1]]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                color='red', linestyle='--', linewidth=4)
        return fig

    def check_shared_sides(self):
        assert (
             isinstance(self.shared_sides, list)
        ), (
             'Arg shared_sides should be a list of lists'
        )
        for i in range(len(self.shared_sides)):
            assert (
                isinstance(self.shared_sides[i][0], list)
            ), (
                f'First item in arg shared_sides, item {i}, is not a list'
            )
            assert (
                isinstance(self.shared_sides[i][1], list)
            ), (
                f'Second item in arg shared_sides, item {i}, is not a list'
            )
            assert (
                len(self.shared_sides[i][0]) == 2
            ), (
                f'First item in arg shared_sides, item {i}, must have 2 elements'
            )
            assert (
                len(self.shared_sides[i][1]) == 2
            ), (
                f'Second item in arg shared_sides, item {i}, must have 2 elements'
            )
            self.check_sides_in_triangles(i,
                                          self.shared_sides[i][0],
                                          self.shared_sides[i][1])
        print('The sides you provided seem to be defined correctly!')
        if len(self.shared_sides) != 10:
            print('WARNING: You should have 10 shared sides'
                           +f' --> currently only {len(self.shared_sides)}')

    def check_sides_in_triangles(self, side, vertices, triangle_id):
        triangle_1 = self.triangles[triangle_id[0]]
        triangle_2 = self.triangles[triangle_id[1]]
        common_vertices = set(triangle_1).intersection(set(triangle_2))
        assert len(common_vertices) == 2, (
            f'Side {side}: Triangles {triangle_id[0]} and {triangle_id[1]} do not share exactly two vertices'
        )
        assert vertices[0] in triangle_1 and vertices[0] in triangle_2, (
            f'Vertex {vertices[0]} is not shared by triangles {triangle_id[0]} and {triangle_id[1]}'
        )
        assert vertices[1] in triangle_1 and vertices[1] in triangle_2, (
            f'Vertex {vertices[1]} is not shared by triangles {triangle_id[0]} and {triangle_id[1]}'
        )
        return True


    def check_initialization(self):
        assert (
             isinstance(self.coordinates, np.ndarray)
        ), (
             'Arg coordinates should be a numpy array'
        )
        assert (
            self.coordinates.shape[1] == 2
        ), (
            'Arg coordinates should have 2 columns'
        )
        assert (
            self.coordinates.shape[0] == 11
        ), (
            'Arg coordinates should have 11 rows'
        )

    def check_triangles(self, triangles=None, triangle_id=None, report=True):
        tol = self.side_length/20
        if triangles is None:
            assert self.triangles is not None, (
                'Define triangles with method define_triangles() or provide them as argument')
            triangles = self.triangles
        if triangle_id is None:
            triangle_id = range(len(triangles))
        assert (
             isinstance(triangles, list)
        ), (
             'Arg triangles should be a list of lists'
        )
        assert (
             isinstance(triangles[0], list)
        ), (
             'First item in arg triangles is not a list'
        )
        for i, tri in enumerate(triangles):
            assert (
                len(tri) == 3
            ), (
                f'Problem with triangle at row {i}: must have 3 vertices'
            )

        # if self.refinement_number == 0:
        #     if len(triangles) != 10:
        #         print('WARNING: You should have 10 triangles'
        #                     +f' --> currently only {len(triangles)}')
        
        height = np.sqrt(self.side_length**2 - (self.side_length/2)**2)
        area = self.side_length*height/2
        for i in range(len(triangles)):
            area_i = self.get_triangle_area(i, triangles)
            assert (
                np.isclose(area_i, area, atol=tol)
            ), (
                f'Triangle {i} does not have the correct area:\n'
                +f'  {area_i:.3e} instead of {area:.3e} (tol = {tol:.2e})'
            )
            for j in range(3):
                side_length = np.linalg.norm(self.coordinates[tri[j]] - self.coordinates[tri[(j+1)%3]])
                assert (
                    np.isclose(side_length, self.side_length, atol=tol)
                ), (
                    f'One of the sides of triangle {i} has incorrect length'
                )
        if report:
            print('All triangles seem to be defined correctly!')