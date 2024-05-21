import numpy as np

class OBJ:
    def __init__(self, filepath):
        self.filepath = filepath
        self.vertices = []
        self.textures = []
        self.normals = []
        self.faces = []
        self.is_triangle_mesh = True  # 默认情况下假设是三角面
        self.load_obj()

    def load_obj(self):
        with open(self.filepath, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    self.vertices.append(self.parse_vertex(line))
                elif line.startswith('vt '):
                    self.textures.append(self.parse_texture(line))
                elif line.startswith('vn '):
                    self.normals.append(self.parse_normal(line))
                elif line.startswith('f '):
                    face = self.parse_face(line)
                    self.faces.append(face)
                    if len(face) == 4:
                        self.is_triangle_mesh = False

    def parse_vertex(self, line):
        parts = line.strip().split()
        return [float(parts[1]), float(parts[2]), float(parts[3])]

    def parse_texture(self, line):
        parts = line.strip().split()
        return [float(parts[1]), float(parts[2])]

    def parse_normal(self, line):
        parts = line.strip().split()
        return [float(parts[1]), float(parts[2]), float(parts[3])]

    def parse_face(self, line):
        parts = line.strip().split()[1:]
        face = []
        for part in parts:
            indices = part.split('/')
            vertex_index = int(indices[0]) if indices[0] else None
            texture_index = int(indices[1]) if len(indices) > 1 and indices[1] else None
            normal_index = int(indices[2]) if len(indices) > 2 and indices[2] else None
            face.append((vertex_index, texture_index, normal_index))
        return face

    # 将读取的数据保存为npz文件是因为npz文件的读取速度更快，当训练数据变多时，文件读写会显著拖累程序的执行速度
    def save_to_npz(self, output_filepath):
        # 将 faces 和 format 转换为适合保存的格式
        faces_array = np.array(self.faces, dtype=object)

        np.savez(
            output_filepath,
            vertices=np.array(self.vertices),
            textures=np.array(self.textures),
            normals=np.array(self.normals),
            faces=faces_array,
            is_triangle_mesh=self.is_triangle_mesh
        )

    def load_from_npz(self, input_filepath):
        data = np.load(input_filepath, allow_pickle=True)
        self.vertices = data['vertices'].tolist()
        self.textures = data['textures'].tolist()
        self.normals = data['normals'].tolist()
        self.faces = data['faces'].tolist()
        self.is_triangle_mesh = data['is_triangle_mesh'].item()


# 示例用法
# obj = OBJ('data//body.obj')
# obj.save_to_npz('data//output_file.npz')
# # 加载并打印数据
# obj.load_from_npz('data//output_file.npz')
# print(obj.vertices)
# print(obj.textures)
# print(obj.normals)
# print(obj.faces)
# print(obj.is_triangle_mesh)