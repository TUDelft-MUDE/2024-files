import numpy as np
import timeit
import matplotlib.pyplot as plt


class Mesh:
    def __init__(self, coordinates, side_length):
        self.coordinates = coordinates
        self.side_length = side_length
        self.triangles = None
        self.shared_sides = None
        self.bar_coordinates = None
        self.kapsalon_coordinates = None
        self.refinement_number = 0
        self.check_initialization()

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

    # def define_triangles(self, triangles):
    #     self.triangles = triangles

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
        self.check_triangles(triangles=triangles, report=False)
        return triangles

    def get_triangle_area(self, triangle_id, triangles):
        triangle = triangles[triangle_id]
        a = self.coordinates[triangle[0]]
        b = self.coordinates[triangle[1]]
        c = self.coordinates[triangle[2]]
        return 0.5*np.abs(np.cross(b-a, c-a))
    
    def get_bar_coordinates(self):
        assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Define with method define_triangles(), or \n'
                +'--> Provide as keyword argument triangles=[[],[],...]\n')
        self.bar_coordinates = np.zeros((len(self.triangles), 2))
        for i, triangle in enumerate(self.triangles):
            a = self.coordinates[triangle[0]]
            b = self.coordinates[triangle[1]]
            c = self.coordinates[triangle[2]]
            centroid = (a + b + c) / 3
            self.bar_coordinates[i] = centroid
        return self.bar_coordinates
    
    def get_kapsalon_coordinates(self):
        assert self.shared_sides is not None, (
                'NO SIDES DEFINED!\n'
                +'--> Define with method define_shared_sides(), or \n'
                +'--> Provide as keyword argument sides=[[[],[]],...]\n')
        self.kapsalon_coordinates = np.zeros((len(self.shared_sides), 2))
        for i, side in enumerate(self.shared_sides):
            triangle_1_id, triangle_2_id = side[1]
            bar_1 = self.get_bar_coordinates()[triangle_1_id]
            bar_2 = self.get_bar_coordinates()[triangle_2_id]
            midpoint = (bar_1 + bar_2) / 2
            self.kapsalon_coordinates[i] = midpoint
        return self.kapsalon_coordinates

    def plot_everything(self):
        fig = self.plot_coordinates()
        ax = fig.get_axes()[0]
        if self.bar_coordinates is not None:
            ax.scatter(self.bar_coordinates[:, 0],
                       self.bar_coordinates[:, 1],
                       label='Bars',
                       marker='s',
                       color='red')
        if self.kapsalon_coordinates is not None:
            ax.scatter(self.kapsalon_coordinates[:, 0],
                       self.kapsalon_coordinates[:, 1],
                       label='Kapsalon Shops',
                       marker='^',
                       color='orange')
        ax.legend()
        ax.set_title('My Plan for the City')
        return fig

    def plot_coordinates(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.coordinates[:, 0],
                   self.coordinates[:, 1],
                   label='Student Houses',
                   color='blue')
        ax.set_title('Triangle Vertices')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        text_offset = 0.1
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
    
    def plot_triangles(self, triangles=None, triangle_id=None):
        if triangles is None:
            assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Define with method define_triangles(), or \n'
                +'--> Provide as keyword argument triangles=[[],[],...]\n'
        )
            triangles = self.triangles
        if triangle_id is None:
            triangle_id = range(len(triangles))
        fig = self.plot_coordinates()
        ax = fig.get_axes()[0]
        if isinstance(triangle_id, int):
            ax.set_title(f'Triangle {triangle_id} (vertices {triangles[triangle_id]})')
            triangle_id = [triangle_id]
        else:
            ax.set_title(f'Triangles plotted: {triangle_id}')
            
        for t_id in triangle_id:
            triangle = triangles[t_id]
            for i in range(3):
                start = self.coordinates[triangle[i]]
                end = self.coordinates[triangle[(i+1)%3]]
                ax.plot([start[0], end[0]], [start[1], end[1]],
                    color='black', linestyle='--', linewidth=2)
                ax.fill(self.coordinates[triangle, 0], 
                    self.coordinates[triangle, 1], 
                    color='green', alpha=0.1)
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

        if self.refinement_number == 0:
            if len(triangles) != 10:
                print('WARNING: You should have 10 triangles'
                            +f' --> currently only {len(triangles)}')
        
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