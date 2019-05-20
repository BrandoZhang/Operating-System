#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>
#include <linux/signal.h>
// #include <linux/sched/signal.h>

MODULE_LICENSE("GPL");

static struct task_struct *task;

struct wait_opts
{
    enum pid_type wo_type;
    int wo_flags;
    struct pid *wo_pid;

    struct siginfo __user *wo_info;
    int __user *wo_stat;
    struct rusage __user *wo_rusage;

    wait_queue_t child_wait;
    int notask_error;
};

extern long _do_fork(unsigned long clone_flags,
                     unsigned long stack_start,
                     unsigned long stack_size,
                     int __user *parent_tidptr,
                     int __user *child_tidptr,
                     unsigned long tls);

extern long do_wait(struct wait_opts *wo);

extern int do_execve(struct filename *,
                     const char __user *const __user *,
                     const char __user *const __user *);

extern struct filename *getname(const char __user *);

const char *IPC_SIG[23] = {  // IPC Signal name with correspoding number in Linux
    "0", "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL",
    "SIGTRAP", "SIGABRT", "SIGBUS", "SIGFPE",
    "SIGKILL", "SIGUSR1", "SIGSEGV", "SIGUSR2",
    "SIGPIPE", "SIGALRM", "SIGTERM", "SIGSTKFLT",
    "SIGCHLD", "SIGCONT", "SIGSTOP", "SIGSTP",
    "SIGTTIN", "SIGTTOU"};

/* function of child process    */
unsigned long DoMyExecve(void *argc)
{
    int flag;
    const char __user *filename = "/media/sf_CSC_3150_r/Ass/CSC3150_Assignment_1/source/program2/test";
    // const char __user *filename = "./test";
    printk("[program2] : child process");
    flag = do_execve(getname(filename), NULL, NULL);  // execute test program
    printk("[program2] : child process return %d\n", flag);
    return 0;
}

/* implement fork function  */
int my_fork(void *argc)
{
    struct wait_opts wo;
    int __user state;  // initialize state variable for wait_opts 
    long pid;

    /* set default sigaction for current process */
    int i;
    struct k_sigaction *k_action = &current->sighand->action[0];
    for (i = 0; i < _NSIG; i++)
    {
        k_action->sa.sa_handler = SIG_DFL;
        k_action->sa.sa_flags = 0;
        k_action->sa.sa_restorer = NULL;
        sigemptyset(&k_action->sa.sa_mask);
        k_action++;
    }

    /* fork a process using do_fork */
    pid = _do_fork(SIGCHLD, (unsigned long)&DoMyExecve, 0, NULL, NULL, 0);

    printk("[program2] : The child process has pid = %ld\n", pid);
    printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);

    /* initialize wait options	*/
    wo.wo_type = PIDTYPE_PID;
    wo.wo_flags = WEXITED;
    wo.wo_pid = find_get_pid(pid);
    wo.wo_info = NULL;
    wo.wo_stat = &state;
    wo.wo_rusage = NULL;

    /* wait until child process terminates */
    do_wait(&wo);
    printk("get %s signal, signal number = %d\n", IPC_SIG[*wo.wo_stat], *wo.wo_stat);

    return 0;
}

static int __init program2_init(void)
{

    printk("[program2] : Module_init\n");

    /* write your code here */

    /* create a kernel thread to run my_fork */
    printk("[program2] : Module_init create kthread start\n");
    task = kthread_create(&my_fork, NULL, "Program2ForkThread");

    /* wake up new thread if ok */
    if (!IS_ERR(task))
    {
        printk("[program2] : Module_init kthread starts\n");
        wake_up_process(task);
    }

    return 0;
}

static void __exit program2_exit(void)
{
    printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
