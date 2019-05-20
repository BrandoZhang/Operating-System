#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"
#define SLEEP_TIME 1000

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;

// Device
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdevp;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);

static int drv_open(struct inode *, struct file *);

static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);

static int drv_release(struct inode *, struct file *);

static long drv_ioctl(struct file *, unsigned int, unsigned long);

// cdev file_operations
static struct file_operations fops = {
        owner: THIS_MODULE,
        read: drv_read,
        write: drv_write,
        unlocked_ioctl: drv_ioctl,
        open: drv_open,
        release: drv_release,
};

// in and out function
void myoutc(unsigned char data, unsigned short int port);

void myouts(unsigned short data, unsigned short int port);

void myouti(unsigned int data, unsigned short int port);

unsigned char myinc(unsigned short int port);

unsigned short myins(unsigned short int port);

unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;  // operator, '+', '-', '*', '/' or 'p'
    int b;  // operand 1
    short c;  // operand 2
} *dataIn;

// get the n-th prime number from base
int prime(int base, short nth) {
    int fnd = 0;
    int i, num, isPrime;

    num = base;
    while (fnd != nth) {
        isPrime = 1;
        num++;
        for (i = 2; i <= num / 2; i++) {
            if (num % i == 0) {
                isPrime = 0;
                break;
            }
        }

        if (isPrime) {
            fnd++;
        }
    }
    return num;
}

// IRQ handler
irqreturn_t IRQ_handler(int irq, void *dev_id) {
    unsigned int irq_counter;
    irq_counter = myini(DMACOUNTADDR);
    irq_counter++;
    myouti(irq_counter, DMACOUNTADDR);
    return IRQ_HANDLED;
}

// Arithmetic function
static void drv_arithmetic_routine(struct work_struct *ws);


// Output and input data to/from DMA
void myoutc(unsigned char data, unsigned short int port) {
    *(volatile unsigned char *) (dma_buf + port) = data;
}

void myouts(unsigned short data, unsigned short int port) {
    *(volatile unsigned short *) (dma_buf + port) = data;
}

void myouti(unsigned int data, unsigned short int port) {
    *(volatile unsigned int *) (dma_buf + port) = data;
}

unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char *) (dma_buf + port);
}

unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short *) (dma_buf + port);
}

unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int *) (dma_buf + port);
}

void init_DMA(void) {
    /* Initialize DMA with 0 or '0' */

    myouti(0, DMASTUIDADDR);
    myouti(0, DMARWOKADDR);
    myouti(0, DMAIOCOKADDR);
    myouti(0, DMAIRQOKADDR);
    myouti(0, DMACOUNTADDR);
    myouti(0, DMAANSADDR);
    myouti(0, DMAREADABLEADDR);
    myouti(0, DMABLOCKADDR);
    myoutc('0', DMAOPCODEADDR);
    myouti(0, DMAOPERANDBADDR);
    myouts(0, DMAOPERANDCADDR);
}

static int drv_open(struct inode *ii, struct file *ff) {
    try_module_get(THIS_MODULE);
    printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
    return 0;
}

static int drv_release(struct inode *ii, struct file *ff) {
    module_put(THIS_MODULE);
    printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
    return 0;
}

static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
    /* Implement read operation for your device */

    int ans, ret;
    ans = myini(DMAANSADDR);  // get answer from DMA buffer
    printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, ans);
    ret = copy_to_user(buffer, &ans, ss);
    if (ret < 0) {
        printk("%s:%s(): ERROR! Fail to copy ans = %d to user buffer.\n", PREFIX_TITLE, __func__, ans);
    }
    myouti(0, DMAANSADDR);  // clean answer
    myouti(0, DMAREADABLEADDR);  // reset readable bit
    return 0;
}

static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
    /* Implement write operation for your device */

    unsigned int IOMode = myini(DMABLOCKADDR);
    printk("%s:%s(): IO Mode is %d\n", PREFIX_TITLE, __func__, IOMode);

    INIT_WORK(work_routine, drv_arithmetic_routine);

    // write data to DMA buffer
    dataIn = (struct DataIn *)buffer;
    myoutc(dataIn->a, DMAOPCODEADDR);
    myouti(dataIn->b, DMAOPERANDBADDR);
    myouts(dataIn->c, DMAOPERANDCADDR);

    // Decide io mode
    if(IOMode) {
        // Blocking IO
        printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);
        schedule_work(work_routine);
        printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
        flush_scheduled_work();
    }
    else {
        // Non-locking IO
        printk("%s,%s(): queue work\n", PREFIX_TITLE, __func__);
        schedule_work(work_routine);
    }

    return 0;
}

static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
    /* Implement ioctl setting for your device */

    int value;
    switch (cmd) {
        case HW5_IOCSETSTUID:
            get_user(value, (int *) arg);  // get value from user space
            myouti(value, DMASTUIDADDR);
            printk("%s:%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, value);
            break;
        case HW5_IOCSETRWOK:
            get_user(value, (int *) arg);  // get value from user space
            myouti(value, DMARWOKADDR);
            printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
            break;
        case HW5_IOCSETIOCOK:
            get_user(value, (int *) arg);  // get value from user space
            myouti(value, DMAIOCOKADDR);
            printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
            break;
        case HW5_IOCSETIRQOK:
            get_user(value, (int *) arg);  // get value from user space
            myouti(value, DMAIRQOKADDR);
            printk("%s:%s(): IRC OK\n", PREFIX_TITLE, __func__);
            break;
        case HW5_IOCSETBLOCK:
            get_user(value, (int *) arg);  // get value from user space
            myouti(value, DMABLOCKADDR);  // change BLOCK bit according to value (0 or 1)
            if (value) {
                printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
            } else {
                printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
            }
            break;
        case HW5_IOCWAITREADABLE:
            /* Wait if not readable */
            while (!myini(DMAREADABLEADDR))
                msleep(SLEEP_TIME);
            put_user(1, (int *) arg);
            printk("%s:%s(): wait readable 1\n", PREFIX_TITLE, __func__);
            break;
        default:
            printk("INVALID CMD!");
            return -1;
    }
    return 0;
}

static void drv_arithmetic_routine(struct work_struct* ws) {
    /* Implement arithmetic routine */

    struct DataIn data;
    int ans;

    data.a = myinc(DMAOPCODEADDR);  // get operator from DMA
    data.b = myini(DMAOPERANDBADDR);  // get operand 1 from DMA
    data.c = myins(DMAOPERANDCADDR);  // get operand 2 from DMA

    switch (data.a) {
        case '+':
            ans = data.b + data.c;
            break;
        case '-':
            ans = data.b - data.c;
            break;
        case '*':
            ans = data.b * data.c;
            break;
        case '/':
            ans = data.b / data.c;
            break;
        case 'p':
            ans = prime(data.b, data.c);
            break;
        default:
            ans = 0;
    }

    myouti(ans, DMAANSADDR);  // write the answer to DMA buffer
    myouti(1, DMAREADABLEADDR);  // set readable bit
    printk("%s:%s(): %d %c %d = %d\n", PREFIX_TITLE, __func__, data.b, data.a, data.c, ans);
}

static int __init init_modules(void) {
    dev_t dev;
    int ret = 0;

    printk("%s:%s(): ............... Start ...............\n", PREFIX_TITLE, __func__);

    /* Register chrdev */
    ret = alloc_chrdev_region(&dev, 0, 1, "mydev");
    if (ret < 0)
    {
        printk("Cannot alloc chrdev\n");
        return ret;
    }

    dev_major = MAJOR(dev);
    dev_minor = MINOR(dev);
    printk("%s:%s(): register chrdev(%d, %d)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);

    /* Init cdev and make it alive */
    dev_cdevp = cdev_alloc();

    cdev_init(dev_cdevp, &fops);
    dev_cdevp->owner = THIS_MODULE;
    ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);
    if (ret < 0) {
        printk("Add chrdev failed\n");
        return ret;
    }

    /* Allocate DMA buffer */
    dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
    if (dma_buf) {
        printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __func__);
    } else {
        printk("%s:%s(): Fail to allocate dma buffer\n", PREFIX_TITLE, __func__);
        return (int) dma_buf;
    }

    /* Allocate work routine */
    work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

    /* Initialize IRQ */
    myouti(0, DMACOUNTADDR);  // initialize irq counter to 0
    ret = request_irq(1, IRQ_handler, IRQF_SHARED, "keyboard_handler", (void *) &IRQ_handler);
    if (ret < 0) {
        printk("%s:%s(): Fail to initialize IRQ handler with code %d\n", PREFIX_TITLE, __func__, ret);
        return ret;
    }
    printk("%s:%s(): request_irq 1 return %d\n", PREFIX_TITLE, __func__, ret);

    init_DMA();

    return 0;
}

static void __exit exit_modules(void) {
    dev_t dev;
    int irq_count;

    dev = MKDEV(dev_major, dev_minor);

    /* Free IRQ */
    free_irq(1, (void *) IRQ_handler);
    irq_count = myini(DMACOUNTADDR);
    printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, irq_count);

    /* Free DMA buffer when exit modules */
    kfree(dma_buf);
    printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

    /* Delete character device */
    cdev_del(dev_cdevp);
    unregister_chrdev_region(dev, 1);
    printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);

    /* Free work routine */
    kfree(work_routine);

    printk("%s:%s(): .............. End ..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
