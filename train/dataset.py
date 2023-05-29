import numpy as np
from torch.utils.data.dataset import Dataset
import natsort, torch, os, scipy
from scipy import ndimage

def generate_position(
    end: int,
    except_list: list,
    start: int = 0,   
):
    my_list = [f"{i:03d}" for i in range(start, end) if i not in except_list]
    print(my_list)


# MicroDL_3D
class MicroDLDataset(Dataset):
    """load microDL"""
    def __init__(
        self, 
        path: str = "/home/yingmuzhi/microDL_3D/_output/retardance2dna_microdl_patches/tiles_128-128_step_128-128", 
        train: bool = True, 
        position: dict={"train":    
                            # ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159']
                            ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019']
                            # ['000']
                        ,"val":      
                            ["021", "022"],} ,
        multimodal: bool = False,
        path2: str = "bulabula",
        path3: str = "",
        path4: str = "",

        single_signal_channel = "phase",
        single_target_channel = "405",

        transform: list = None,
    ) -> None:
        """
        intro:
            parse dataset in MDL.
        """
        super().__init__()

        # transform is not none, then add self.transform
        self.transform = transform
        
        self.data_path = path
        self.train = train

        self.multimodal = multimodal

        # store file name
        img_input = []
        img_label = []
        # if using multimodal
        if multimodal:
            img_input2 = []
            img_input3 = []
            img_input4 = []

        # store file path
        self.input = []
        self.label = []
        # if using multimodal
        if multimodal:
            self.input2 = []
            self.input3 = []
            self.input4 = []


        if self.train:
            for pos in position["train"]:
                img_input_pos = [i for i in os.listdir(self.data_path) if single_signal_channel in i  and "p{}".format(str(pos)) in i]
                img_label_pos = [i for i in os.listdir(self.data_path) if single_target_channel in i    and "p{}".format(str(pos)) in i]
                img_input.extend(img_input_pos)
                img_label.extend(img_label_pos)

                # if using multimodal
                if multimodal:
                    # retardance
                    self.data_path2 = path2
                    img_input_pos2 = [i for i in os.listdir(self.data_path2) if "Retardance" in i and "p{}".format(str(pos)) in i] 
                    img_input2.extend(img_input_pos2)

                    # o_x
                    self.data_path3 = path3
                    img_input_pos3 = [i for i in os.listdir(self.data_path3) if "Orientation" in i and "p{}".format(str(pos)) in i] 
                    img_input3.extend(img_input_pos3)
                    
                    # o_y
                    self.data_path4 = path4
                    img_input_pos4 = [i for i in os.listdir(self.data_path4) if "Orientation" in i and "p{}".format(str(pos)) in i] 
                    img_input4.extend(img_input_pos4)

                
        else:
            for pos in position["val"]:
                files = os.listdir(path)
                img_input_pos = [file for file in files if single_signal_channel in file  and "p{}".format(str(pos)) in file] # im_c001_z019_t000_p003_r0-128_c384-512_sl0-32
                img_label_pos = [file for file in files if single_target_channel in file    and "p{}".format((str(pos))) in file] # im_c000_z019_t000_p003_r0-128_c384-512_sl0-32
                img_input.extend(img_input_pos)
                img_label.extend(img_label_pos)

                # if using multimodal
                if multimodal:
                    self.data_path2 = path2
                    img_input_pos2 = [i for i in os.listdir(self.data_path2) if "Retardance" in i and "p{}".format(str(pos)) in i] 
                    img_input2.extend(img_input_pos2)

                     # o_x
                    self.data_path3 = path3
                    img_input_pos3 = [i for i in os.listdir(self.data_path3) if "Orientation" in i and "p{}".format(str(pos)) in i] 
                    img_input3.extend(img_input_pos3)
                    
                    # o_y
                    self.data_path4 = path4
                    img_input_pos4 = [i for i in os.listdir(self.data_path4) if "Orientation" in i and "p{}".format(str(pos)) in i] 
                    img_input4.extend(img_input_pos4)



        self.input: list = [os.path.join(path, i) for i in img_input]
        self.label: list = [os.path.join(path, i) for i in img_label]
        self.input = natsort.natsorted(self.input)
        self.label = natsort.natsorted(self.label)

        # if using multimodal
        if multimodal:
            self.input2 = [os.path.join(path2, i) for i in img_input2]
            self.input2 = natsort.natsorted(self.input2)

            self.input3 = [os.path.join(path3, i) for i in img_input3]
            self.input3 = natsort.natsorted(self.input3)

            self.input4 = [os.path.join(path4, i) for i in img_input4]
            self.input4 = natsort.natsorted(self.input4)
        pass

    def __getitem__(self, index):
        input: torch.Tensor = None
        label: torch.Tensor = None

        # load npy
        input = np.load(self.input[index])
        label = np.load(self.label[index])

        # # print(type(input.shape))
        # if list(label.shape) == [32, 256, 256]:
        #     print("h")
        # else:
        #     print("error")
        
        # # resize
        # input = Resizer([1,0.5,0.5])((input))
        # label = Resizer([1,0.5,0.5])((label))


        # expand dimensions
        input = np.expand_dims(input, 0)
        label = np.expand_dims(label, 0)


        # change dtype
        input = input.astype("float32")
        label = label.astype("float32") 

        if self.multimodal:
        # if using multimodal
            input2 = np.load(self.input2[index])
            input2 = np.expand_dims(input2, 0)
            input2 = input2.astype("float32")   

            input3 = np.load(self.input3[index])
            input3 = np.expand_dims(input3, 0)
            input3 = input3.astype("float32")

            input4 = np.load(self.input4[index])
            input4 = np.expand_dims(input4, 0)
            input4 = input4.astype("float32")
            # stack
            input = np.stack((input, input2, input3, input4), axis=1).squeeze(0)

        #transform
        # if self.transform is not None:
        if self.transform:
            X_tensor, Y_tensor = self.transform(input, label)
        else:
            X_tensor = torch.tensor(input)
            Y_tensor = torch.tensor(label)

        return X_tensor, Y_tensor
    
    def __len__(self):
        return len(self.label)

    @staticmethod
    def collate_fn(batch):
        # pack images
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    # make every images the same size as [channel, width, height]
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


    
class Resizer(object):
    def __init__(self, factors):
        """
        factors - tuple of resizing factors for each dimension of the input array"""
        self.factors = factors

    def __call__(self, x):
        return scipy.ndimage.zoom(x, (self.factors), mode='nearest')

    def __repr__(self):
        return 'Resizer({:s})'.format(str(self.factors)) 




a = np.load("/home/yingmuzhi/microDL_2_0/data_retardance2actin/output/crop_img_568_p021_z000to031_row0000to0256_col0512to0768.npy")
pass


# test
if __name__ == "__main__":
    # generate_position(start=10, end=32, except_list=[21, 22])

    # path1 = "/home/yingmuzhi/microDL_2_0/data_phase2nuclei/output"
    # path2 = "/home/yingmuzhi/microDL_2_0/data_retardance2nuclei/output"
    # path3 = "/home/yingmuzhi/microDL_2_0/data_orientation_x2nuclei/output"
    # path4 = "/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/output"
    # MicroDLDataset(
    #     path=path1,
    #     train=False,
    #     multimodal = True,
    #     path2 = path2,
    #     path3= path3,
    #     path4= path4
    # )[21]

    from transform import TransMDLTransform

    train_path1 = "/home/yingmuzhi/microDL_2_0/data_retardance2actin/output"
    train_dataset = MicroDLDataset(path = train_path1, train=False, multimodal=False, single_signal_channel="Retardance", single_target_channel="568",
        transform=TransMDLTransform(train_or_eval=False),
        )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=1)
    for temp in train_loader:
        pass
    pass
