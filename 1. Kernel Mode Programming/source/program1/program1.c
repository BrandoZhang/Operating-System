#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]) {

    int status;
    /* fork a child process */
    printf("Process start to fork\n");
    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork error");
        exit(EXIT_FAILURE);
    } else {

        // Child process
        if (pid == 0) {
            printf("I'm the child process, my pid = %d\n", getpid());

            int i;
            char *arg[argc];

            for (i = 0; i < argc - 1; i++) {
                // Get the command line variable except the first one
                arg[i] = argv[i + 1];
            }
            arg[argc - 1] = NULL;

            /* execute test program */
            printf("Child process start to execute the program\n");
            execve(arg[0], arg, NULL);

            // Check if the child process is replaced by new program
            printf("Continue to run original child process!\n");

            perror("execve");
            exit(EXIT_FAILURE);
        }

            // Parent process
        else {
            printf("I'm the parent process, my pid = %d\n", getpid());

            /* wait for child process terminates */
//            wait(&status);
            waitpid(pid, &status, WUNTRACED);
            printf("Parent process receiving the SIGCHLD signal\n");

            /* check child process'  termination status */
            if (WIFEXITED(status)) {
                printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
            } else if (WIFSIGNALED(status)) {
                int child_term_sign = WTERMSIG(status);
                switch (child_term_sign) {
                    case SIGABRT: {
                        printf("child process get SIGABRT signal\n");
                        printf("child process is terminated by abort signal\n");
                        break;
                    }
                    case SIGALRM: {
                        printf("child process get SIGALRM signal\n");
                        printf("child process is terminated by alarm signal\n");
                        break;
                    }
                    case SIGBUS: {
                        printf("child process get SIGBUS signal\n");
                        printf("child process is terminated by bus signal\n");
                        break;
                    }
                    case SIGFPE: {
                        printf("child process get SIGFPE signal\n");
                        printf("child process is terminated by floating signal\n");
                        break;
                    }
                    case SIGHUP: {
                        printf("child process get SIGHUP signal\n");
                        printf("child process is terminated by hangup signal\n");
                        break;
                    }
                    case SIGILL: {
                        printf("child process get SIGILL signal\n");
                        printf("child process is terminated by illegal instruction signal\n");
                        break;
                    }
                    case SIGINT: {
                        printf("child process get SIGINT signal\n");
                        printf("child process is terminated by interrupt signal\n");
                        break;
                    }
                    case SIGKILL: {
                        printf("child process get SIGKILL signal\n");
                        printf("child process is terminated by kill signal\n");
                        break;
                    }
                    case SIGPIPE: {
                        printf("child process get SIGPIPE signal\n");
                        printf("child process is terminated by pipe signal\n");
                        break;
                    }
                    case SIGQUIT: {
                        printf("child process get SIGQUIT signal\n");
                        printf("child process is terminated by quit signal\n");
                        break;
                    }
                    case SIGSEGV: {
                        printf("child process get SIGSEGV signal\n");
                        printf("child process is terminated by invalid memory reference.\n");
                        break;
                    }
                    case SIGTERM: {
                        printf("child process get SIGTERM signal\n");
                        printf("child process is terminated by termination signal\n");
                        break;
                    }
                    case SIGTRAP: {
                        printf("child process get SIGTRAP signal\n");
                        printf("child process is terminated by trap signal\n");
                        break;
                    }
                    default: {
                        printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
                        break;
                    }
                }
                printf("CHILD EXECUTION FAILED!!\n");

            } else if (WIFSTOPPED(status)) {
                int child_stop_signal = WSTOPSIG(status);
                switch (child_stop_signal) {
                    case SIGSTOP: {
                        printf("child process get SIGSTOP signal\n");
                        printf("child process stopped\n");
                        break;
                    }
                    default: {
                        printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
                        break;
                    }
                }

                printf("CHILD PROCESS STOPPED\n");
            } else {
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(EXIT_SUCCESS);
        }
    }

    return 0;
}
