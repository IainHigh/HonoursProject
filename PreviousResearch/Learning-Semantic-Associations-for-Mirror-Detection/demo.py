import os, torch, numpy, ntpath, tqdm
from PIL import Image
from model.architecture import Architecture

checkpoint_dir = "/home/iain/Desktop/HonoursProject/PreviousResearch/Learning-Semantic-Associations-for-Mirror-Detection/checkpoints-20230927T155630Z-001/checkpoints/u_train_on_pmd_lr5eN4it40k.pt"
input_dir = (
    "/home/iain/Desktop/HonoursProject/Datasets/PMD_benchmark_combined/mine/test/image"
)
output_dir = "/home/iain/Desktop/HonoursProject/result"
crf = True

print(
    "Load checkpoints:",
    checkpoint_dir,
    "\nInput:",
    input_dir,
    "\nOutput:",
    output_dir,
    "\nCRF:",
    crf,
)

model = Architecture(training=False)
model.load_state_dict(torch.load(checkpoint_dir, map_location="cpu"))
model.eval()

images = [input_dir]
if os.path.isdir(input_dir):
    images = os.listdir(input_dir)
    images = [
        os.path.join(input_dir, x)
        for x in images
        if x[-4::] in [".jpg", ".png", ".JPG", ".PNG", "jpeg"]
    ]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for imgpath in tqdm.tqdm(images):
    torch_img = torch.from_numpy(
        numpy.array(Image.open(imgpath), dtype=numpy.uint8)
    )  # H, W, C
    if len(torch_img.shape) == 2:
        torch_img = torch.stack([torch_img, torch_img, torch_img], dim=2)
    if len(torch_img.shape) == 3 and torch_img.shape[-1] >= 4:
        torch_img = torch_img[:, :, 0:3]
    img = torch_img.numpy()

    size = torch_img.shape[0:2]
    torch_img = torch.nn.functional.interpolate(
        torch_img.permute(2, 0, 1).unsqueeze(0), size=(384, 384), mode="nearest"
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    torch_img = (torch_img / 255.0 - mean) / std
    out = model(torch_img)["final"].cpu().detach()
    out = torch.sigmoid(
        torch.nn.functional.interpolate(out, size=size, mode="bilinear")
    )
    out = (out.numpy() * 255.0).astype(numpy.uint8)[0, 0]

    if crf:
        from crf import crf_refine

        out = crf_refine(img.astype(numpy.uint8), out)

    Image.fromarray(out.astype(numpy.uint8)).save(
        os.path.join(output_dir, ntpath.basename(imgpath).replace(".jpg", ".png"))
    )
    # print("running ", imgpath, end="\r")
print("Done", "see '%s' for the output" % output_dir)
