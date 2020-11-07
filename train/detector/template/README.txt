------------------------------------------

      Maixhub 目标检测训练结果 使用说明

------------------------------------------

文件说明:

* boot.py: 在 maixpy 上运行的代码
* *.kmodel 或者 *.smodel: 训练好的模型文件（ smodel 是加密模型 ）
* labels.txt: 分类标签
* startup.jpg: 启动图标
* report.jpg: 训练报告,包括了损失和准确度报告等
* warning.txt: 训练警告信息，如果有这个文件，务必阅读， 里面的原因可能会导致训练精度低

使用方法:

0. 按照文档(maixpy.sipeed.com)更新到最新的固件
   如果新固件出现bug,可以使用这个固件测试(选择minimum_with_ide_support.bin): 
   https://dl.sipeed.com/MAIX/MaixPy/release/master/maixpy_v0.5.1_124_ga3f708c
1. 准备一张 SD 卡, 将本目录下的文件拷贝到 SD 卡根目录
2. SD 卡插入开发板
3. 开发板上电启动
4. 摄像头对准训练的物体,
       屏幕左上角会显示 物体标签 和 概率
       屏幕左下角会显示 运行模型消耗的时间,单位是毫秒

如果没有 SD 卡:

* 按照 maixpy.sipeed.com 文档所述的方式, 将模型烧录到 flash
* 修改 boot.py 的 main 函数调用的参数: model 在 flash 中的地址
* 其它资源文件,比如 startup.jpg 可以通过工具发送到开发板的文件系统,或者不管, 没有会自动跳过显示
* 运行 boot.py
* 如果以上的步骤您不理解,那么应该先完整按照 maixpy.sipeed.com 的文档学习一遍使用方法就会了


问题反馈: 
   关于 MaixPy 的问题请到这里提问,提问前搜一下是否有相同问题提出过: https://github.com/sipeed/MaixPy/issues
   maixhub 相关问题请邮件: support@sipeed.com, 每天邮件很多,注意邮件格式很重要,不然可能得不到及时回复请谅解
        邮件标题: [maixhub][故障/建议] 标题内容,简洁描述问题 而 不是 "我需要帮助" "为什么用不了了" 这样的问题
        邮件内容: 如果是出现使用问题或者bug,为了更快更好的帮您解决问题, 请务必写好 错误现象, 详细 的复现过程
   也可以到 bbs.sipeed.com 进行讨论

------------------------------------------

Maixhub Object detection Training Results Instructions for Use

------------------------------------------

File Description:

* boot.py: code to run on maixpy
* *.kmodel or *.smodel: training model file (smodel is an encrypted model)
* labels.txt: category labels
* startup.jpg: startup icon
* report.jpg: training report, including loss and accuracy reports, etc.
* warning.txt: training warning message, if you have this file, be sure to read it as it may cause training loss!

Usage :

0. Follow the documentation(maixpy.sipeed.com) to update to the latest firmware
   If the new firmware is buggy, you can use this firmware to test it (choose minimum_with_ide_support.bin): 
   https://dl.sipeed.com/MAIX/MaixPy/release/master/maixpy_v0.5.1_124_ga3f708c
1) Prepare an SD card, copy the files from this directory to the root of the SD card.
2. insert SD card into development board
3. Development board power-up
4. Aim the camera at the training object,
       The top left corner of the screen will show Object labels and probability.
       The bottom left corner of the screen shows the time in milliseconds it takes to run the model.

If you do not have an SD card:

* Burn the model to flash as described in the documentation（maixpy.sipeed.com）
* Modify the parameters of the main boot.py call: model's address in flash.
* Other resource files, such as startup.jpg can be sent to the development board through the tool file system, or no matter, no will automatically skip the display
* Run boot.py
* If you don't understand the above steps, then you should follow the maixpy.sipeed.com documentation to learn how to use it!


Question Feedback: 
   Questions about MaixPy go here( https://github.com/sipeed/MaixPy/issues ) to ask questions,Search before asking if the same question has been asked
   maixhub related questions please mail: support@sipeed.com, every day a lot of mail, pay attention to the mail format is very important, otherwise you may not be able to reply in time, please understand!
        Subject: [maixhub][Troubleshooting/suggestions] The subject of the message, a concise description of the problem, not a question like "I need help" or "Why is it not working".
        Email content: If there is a problem or bug, please make sure to include the error phenomenon and a detailed reproduction process in order to help you solve the problem faster and better.
   You can also discuss at bbs.sipeed.com




