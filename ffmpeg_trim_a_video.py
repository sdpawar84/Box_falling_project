import os

folder_path = "/home/inqedge/Downloads/Object_falling_Videos/"

fr = open("/home/inqedge/Downloads/Object_falling_Videos/abc.txt", "r")
for line in fr:
    try:
        fn, st_time, en_time = [i.strip() for i in line.split("\t")]
    except:
        continue
    file_path = os.path.join(folder_path, fn)
    ot_path = os.path.join(folder_path, "output", f"output_{fn}")
    cmd = f"ffmpeg -ss {st_time} -i {file_path} -t {en_time} -c:v copy -c:a copy {ot_path}"
    print(cmd)
    os.system(cmd)
