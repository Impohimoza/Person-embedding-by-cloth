Person-embedding-by-cloth
=================================
Данный репозиторий предназначен для обучения моделей глубокого обучения, предназначенных для извлечения векторов признаков из изображений людей по их одежде.

Installation
---------------
Убедитесь, что `conda <https://www.anaconda.com/distribution/>`_ установлена.

.. code-block:: bash

    # перейдите в нужную вам директорию и клонируйте этот репозиторий.
    git clone https://github.com/Impohimoza/Person-embedding-by-cloth.git

    # создать среду
    cd Person-embedding-by-cloth
    conda create --name person python=3.13
    conda activate person

    # установить зависимости
    pip install -r requirements.txt

    # Установите torch и torchvision (выберите подходящую версию CUDA для вашего компьютера)
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


Начало работы
-------------

1. Import ``clothclassify``

.. code-block:: python

    import torch
    
    import clothclassify

2. Загрузить data manager

.. code-block:: python

    datamanager = clothclassify.data.ImageDataManager(
        root="data",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_val=100,
        transforms=["random_flip", "random_crop"]
    )

3. Построить модель и оптимизатор

.. code-block:: python

    model = clothclassify.models.build_model(
        'mobilenetv2_x1_4',
        751,
        'triplet',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

4. Построить engine 

.. code-block:: python

    engine = clothclassify.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer
    )

5. Запустите обучение и тест

.. code-block:: python

    engine.run(
        'log/' + f"{type(model).__name__}",
        max_epoch=10,
        start_epoch=0,
        print_freq=1,
        fixbase_epoch=5,
        ranks=[1, 2],
        eval_freq=2
    )

Models
------
- `MobileNetV2 <https://arxiv.org/abs/1801.04381>`_
- `HACNN <https://arxiv.org/abs/1802.08122>`_
- `OSNet <https://arxiv.org/abs/1905.00953>`_
- `OSNet-AIN <https://arxiv.org/abs/1910.06827>`_

Список литературы
-----------------
- `Siamese Neural Networks for One-shot Image Recognition <https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf>`_
- `Person Re-Identification <https://arxiv.org/abs/2204.13158>`_
- `Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch <https://arxiv.org/abs/1910.10093>`_
- `Deep Transfer Learning for Person Re-identification <https://arxiv.org/abs/1611.05244>`_