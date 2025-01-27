def partition_rectangle(rect_width, rect_height, m, n, output_file):
    sub_width = m
    sub_height = n

    with open(output_file, mode='w', newline='') as file:
        for j in range(int(rect_width / sub_width)):
            for i in range(int(rect_height / sub_height)):
                x = j * sub_width
                y = i * sub_height
                edge_x = sub_width
                edge_y = sub_height
                formatted_row = f"{x} {y} {edge_x} {edge_y} : x, y , edge_x, edge_y\n"
                file.write(formatted_row)

if __name__ == "__main__":
    rect_width = 25
    rect_height = 57
    m = 8
    n = 8
    output_file = "cell_positions.csv"
    partition_rectangle(rect_width, rect_height, m, n, output_file)
    print(f"Partitioned rectangle saved to {output_file}")