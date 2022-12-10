# -*- coding: utf-8 -*-
import time,os
import tarfile,pickle,shutil
import PIL.Image
# 文件存储路径
# outdir = 'D:\DeepLearning\Caffe_DIGIST\DataSets\cifar-10_ok'
outdir = '/media/user/data/meta_gan/Matching-DAGAN-1wayKshot/coarse-data'
# 设置图片文件后缀名
file_extension = 'png'
def uncompressData():
    filename = 'cifar-10-python.tar.gz'
    filepath = os.path.join(outdir, filename)
    assert os.path.exists(filepath), 'Expected "%s" to exist' % filename
    # tarfile是一个解压库
    if not os.path.exists(os.path.join(outdir, 'cifar-10-batches-py')):
        # print "Uncompressing file=%s ..." % filename
        with tarfile.open(filepath) as tf:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, outdir)

def uncompressData_cifar100():
    filename = 'cifar-100-python.tar.gz'
    filepath = os.path.join(outdir, filename)
    assert os.path.exists(filepath), 'Expected "%s" to exist' % filename

    if not os.path.exists(os.path.join(outdir, 'cifar-100-python')):
        # print "Uncompressing file=%s ..." % filename
        with tarfile.open(filepath) as tf:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, outdir)

def processData():
    label_filename = 'batches.meta'
    label_filepath = os.path.join(outdir, 'cifar-10-batches-py', label_filename)
    # 打开二进制文件
    with open(label_filepath, 'rb') as infile:
        pickleObj = pickle.load(infile)
        label_names = pickleObj['label_names']

    for phase in 'train', 'test':
        dirname = os.path.join(outdir, phase)
        mkdir_safely(dirname, clean=True)
        with open(os.path.join(dirname, 'labels.txt'), 'w') as outfile:
            for name in label_names:
                outfile.write('%s\n' % name)

    for filename, phase in [
        ('data_batch_1', 'train'),
        ('data_batch_2', 'train'),
        ('data_batch_3', 'train'),
        ('data_batch_4', 'train'),
        ('data_batch_5', 'train'),
        ('test_batch', 'test'),
    ]:
        filepath = os.path.join(outdir, 'cifar-10-batches-py', filename)
        assert os.path.exists(filepath), 'Expected "%s" to exist' % filename
        # 开始提取图片
        extractData(filepath, phase, label_names)

def processData_cifar100():
    label_filename = 'meta'
    label_filepath = os.path.join(outdir, 'cifar-100-python', label_filename)
    with open(label_filepath, 'rb') as infile:
        pickleObj = pickle.load(infile)
        fine_label_names = pickleObj['fine_label_names']
        coarse_label_names = pickleObj['coarse_label_names']

    for level, label_names in [
        ('fine', fine_label_names),
        ('coarse', coarse_label_names),
    ]:
        dirname = os.path.join(outdir, level)
        # mkdir_safely(dirname, clean=True)
        mkdir_safely(dirname, clean=False)
        with open(os.path.join(dirname, 'labels.txt'), 'w') as outfile:
            for name in label_names:
                outfile.write('%s\n' % name)

    for filename, phase in [
        ('train', 'train'),
        ('test', 'test'),
    ]:
        filepath = os.path.join(outdir, 'cifar-100-python', filename)
        assert os.path.exists(filepath), 'Expected "%s" to exist' % filename

        extractData_cifar100(filepath, phase, fine_label_names, coarse_label_names)

def extractData(input_file, phase, label_names):
    """
    Read a pickle file at input_file and output images
    Arguments:
    input_file -- the input pickle file
    phase -- train or test
    label_names -- a list of strings
    """
    # print 'Extracting images file=%s ...' % input_file

    # 读取包文件（二进制）
    with open(input_file, 'rb') as infile:
        pickleObj = pickle.load(infile)
        # print 'Batch -', pickleObj['batch_label']
        data = pickleObj['data']
        # 包解析
        assert data.shape == (10000, 3072), 'Expected data.shape to be (10000, 3072), not %s' % (data.shape,)
        count = data.shape[0]
        labels = pickleObj['labels']
        assert len(labels) == count, 'Expected len(labels) to be %d, not %d' % (count, len(labels))
        filenames = pickleObj['filenames']
        assert len(filenames) == count, 'Expected len(filenames) to be %d, not %d' % (count, len(filenames))

    data = data.reshape((10000, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))

    output_dir = os.path.join(outdir, phase)
    mkdir_safely(output_dir)
    with open(os.path.join(output_dir, '%s.txt' % phase), 'a') as outfile:
        for index, image in enumerate(data):
            # 创建文件夹
            dirname = os.path.join(output_dir, label_names[labels[index]])
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # 获取图片名称
            filename = filenames[index]
            ext = os.path.splitext(filename)[1][1:].lower()
            if ext != file_extension:
                filename = '%s.%s' % (os.path.splitext(filename)[0],file_extension)
            filename = os.path.join(dirname, filename)

            # 保存图片
            PIL.Image.fromarray(image).save(filename)
            outfile.write('%s %s\n' % (filename, labels[index]))


def extractData_cifar100(input_file, phase, fine_label_names, coarse_label_names):
    # Read a pickle file at input_file and output as images
    #
    # Arguments:
    # input_file -- a pickle file
    # phase -- train or test
    # fine_label_names -- mapping from fine_labels to strings
    # coarse_label_names -- mapping from coarse_labels to strings
    # print 'Extracting images file=%s ...' % input_file

    with open(input_file, 'rb') as infile:
        pickleObj = pickle.load(infile,encoding='latin1')
        data = pickleObj['data']
        assert data.shape[1] == 3072, 'Unexpected data.shape %s' % (data.shape,)
        count = data.shape[0]
        fine_labels = pickleObj['fine_labels']
        assert len(fine_labels) == count, 'Expected len(fine_labels) to be %d, not %d' % (count, len(fine_labels))
        coarse_labels = pickleObj['coarse_labels']
        assert len(coarse_labels) == count, 'Expected len(coarse_labels) to be %d, not %d' % (
            count, len(coarse_labels))
        filenames = pickleObj['filenames']
        assert len(filenames) == count, 'Expected len(filenames) to be %d, not %d' % (count, len(filenames))

    data = data.reshape((count, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))

    fine_to_coarse = {}  # mapping of fine labels to coarse labels

    # fine_dirname = os.path.join(outdir, 'fine', phase)
    fine_dirname = os.path.join(outdir, 'fine', 'Total')
    os.makedirs(fine_dirname)
    coarse_dirname = os.path.join(outdir, 'coarse', 'Total')

# coarse_dirname = os.path.join(outdir, 'coarse', phase)
    os.makedirs(coarse_dirname)
    # with open(os.path.join(outdir, 'fine', '%s.txt' % phase), 'w') as fine_textfile, \
    #         open(os.path.join(outdir, 'coarse', '%s.txt' % phase), 'w') as coarse_textfile:
    with open(os.path.join(outdir, 'fine', '%s.txt' % phase), 'w') as fine_textfile, \
            open(os.path.join(outdir, 'coarse', '%s.txt' % phase), 'w') as coarse_textfile:
        for index, image in enumerate(data):
            # Create the directory
            fine_label = fine_label_names[fine_labels[index]]
            dirname = os.path.join(fine_dirname, fine_label)
            mkdir_safely(dirname)

            # Get the filename
            filename = filenames[index]
            ext = os.path.splitext(filename)[1][1:].lower()
            if ext != file_extension:
                filename = '%s.%s' % (os.path.splitext(filename)[0], self.file_extension)
            filename = os.path.join(dirname, filename)

            # Save the image
            PIL.Image.fromarray(image).save(filename)
            fine_textfile.write('%s %s\n' % (filename, fine_labels[index]))
            coarse_textfile.write('%s %s\n' % (filename, coarse_labels[index]))

            if fine_label not in fine_to_coarse:
                fine_to_coarse[fine_label] = coarse_label_names[coarse_labels[index]]
    # Create the coarse dataset with symlinks  windows下无法使用软连接
    # for fine, coarse in fine_to_coarse.iteritems():
    #     mkdir_safely(os.path.join(coarse_dirname, coarse))
    #     os.symlink(
    #         # Create relative symlinks for portability
    #         os.path.join('..', '..', '..', 'fine', phase, fine),
    #         os.path.join(coarse_dirname, coarse, fine)
    #     )

# 安全的新建目录方法
def mkdir_safely(d, clean=False):
    if os.path.exists(d):
        if clean:
            shutil.rmtree(d)
        else:
            return
    os.mkdir(d)

if __name__ == '__main__':
    start = time.time()
    # # print 'Extract Start!'
    # if os.path.exists(outdir + '\\finish'):
    #     print(outdir + '\\finish')
    #     # print 'File Extracting has already done.'
    # else:
    #     uncompressData_cifar100()
    #     processData_cifar100()
    #     mkdir_safely(outdir + '\\finish')
    # # print 'Done after %s secends.' % (time.time() - start)
    uncompressData_cifar100()
    processData_cifar100()
    mkdir_safely(outdir + '\\finish')
