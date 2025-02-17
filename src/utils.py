class Utils():

    def __init__(self):
        pass

    def encode_state(ugv_position, target_position, grid_size):

        ugv_position_index = ugv_position[0] * 5 + ugv_position[1]
        ugv_orientation_index = ugv_position[2] // 90
        ugv_index = ugv_position_index * 4 + ugv_orientation_index
        target_index = target_position[0] * 5 + target_position[1]
        combined_index = target_index + (grid_size**2) * ugv_index
        combined_index = int(combined_index)

        return combined_index

    def decode_state(encoded_state, grid_size):

        ugv_index = encoded_state // (grid_size**2)

        ugv_position_index = ugv_index // 4
        ugv_orientation_index = ugv_index % 4

        ugv_x = ugv_position_index // grid_size
        ugv_y = ugv_position_index % grid_size
        ugv_orientation = ugv_orientation_index * 90

        target_index = encoded_state % (grid_size**2)
        target_x = target_index // grid_size
        target_y = target_index % grid_size

        ugv_position = (ugv_x, ugv_y, ugv_orientation)
        target_position = (target_x, target_y)

        return ugv_position, target_position