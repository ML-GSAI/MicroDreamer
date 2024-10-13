import os
base_dir='./logs'
output_file = "./clip_similarity.txt"
ext_name = ".obj"
ft = "+z" # front direction

def mesh_render(mesh_path,ft='-y',save_path=''):
    import kiui
    from kiui.render import GUI
    import tqdm
    import argparse
    from PIL import Image
    import numpy as np
    import os
    parser = argparse.ArgumentParser()
    # parser.add_argument('mesh', type=str, help="path to mesh (obj, glb, ...)")
    parser.add_argument('--pbr', action='store_true', help="enable PBR material")
    parser.add_argument('--envmap', type=str, default=None, help="hdr env map path for pbr")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth', 'pbr'], help="rendering mode")
    parser.add_argument('--W', type=int, default=512, help="GUI width")
    parser.add_argument('--H', type=int, default=512, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=49, help="default GUI camera fovy")
    parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=8, help="number of images to render from different azimuths")

    os.makedirs(save_path,exist_ok=True)

    opt = parser.parse_args()
    opt.mesh=mesh_path
    opt.force_cuda_rast=True
    opt.wogui = True
    opt.front_dir = ft
    opt.ssaa=1
    gui = GUI(opt)
    elevation = [opt.elevation]
    azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=False)
    for ele in elevation:
        for ii,azi in enumerate(azimuth):
            gui.cam.from_angle(ele, azi)
            gui.need_update = True
            gui.step()
            image = (gui.render_buffer * 255).astype(np.uint8)
            image = Image.fromarray(image)
            
            img_pt=save_path
            os.makedirs(f'{img_pt}',exist_ok=True)
            image.save(f'{img_pt}/{ii}.png')

def gen():
    cnt=0
    os.makedirs('./tmp',exist_ok=True)
    for file in sorted(os.listdir(base_dir)):
        if file.endswith('rgba'+ext_name):
            cnt+=1
            filename=file.split(ext_name)[0]
            mesh_render(os.path.join(base_dir,file),ft=ft,save_path=f'./tmp/{filename}')
    print(cnt)

gen()

from clip_sim import cal_clip_sim
def cal_metrics():
    test_dirs='./tmp'
    sims=[]
    with open(output_file,'w') as f:
        for file in sorted(os.listdir(test_dirs)):
            pt=os.path.join(test_dirs,file)
            if not os.path.isdir(pt):
                continue
            
            ref_img=os.path.join('./test_data',file+'.png')
            print(ref_img)
            novel=[os.path.join(pt,f'{i}.png') for i in range(8)]
            sim=cal_clip_sim(ref_img,novel)
            # print(sim)
            sims.append(sim)
            f.write(f"{file}: {sim}\n")
        print(sum(sims)/len(sims))
        f.write(f"average: {sum(sims)/len(sims)}\n")


cal_metrics()

