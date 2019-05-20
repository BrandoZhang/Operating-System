#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <pthread.h>

#define ROW 10
#define COLUMN 50
#define LOG_LENGTH 15
#define NUM_LOG 9
#define REFRESH_TIME 60000

enum status {
    NORMAL, QUIT, WIN, LOSE
};
status game_status = NORMAL;

pthread_mutex_t map_mutex;  // mutex for modifying map
pthread_cond_t log_complete;  // condition of adding frog on logs

struct Node {
    int x, y;  // x represents the ROW# while y represents the COLUMN#
    bool on_log;  // true if on a log

    Node(int _x, int _y) : x(_x), y(_y) {};

    Node() {};
} frog;

char map[ROW + 10][COLUMN];

/*  Determine a keyboard is hit or not. If yes, return 1. If not, return 0. */
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);

    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);

    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

/*  Check moving direction according to ROW#, true if it moves towards right */
bool direction_is_left(int row_num) {
    return row_num % 2 != 0;
}

/*  True if there is a log  */
bool on_log(Node *node) {
    return map[node->x][node->y] == '=';
}

/*  Change on_log flag if the node is on a log   */
void move_if_on_log(Node *node) {
    if (on_log(node)) node->on_log = true;
    else node->on_log = false;
}

/*  Control logs' movement  */
void *logs_move(void *thread_id) {
    int i, j;
    long tid = (long) thread_id;
    srand((unsigned) time(NULL));
    usleep(rand() % REFRESH_TIME);
    int init_pos = rand() % (COLUMN - 1);
    while (game_status == NORMAL) {
//        usleep(rand() % 2000);  // revert this command to make logs moving at different speed
        usleep(REFRESH_TIME * 2);

        /*  Move the logs  */
        if (direction_is_left(tid + 1)) {
            init_pos--;
            if (init_pos < 0) init_pos += COLUMN - 1;
        } else {
            init_pos = (init_pos + 1) % (COLUMN - 1);
        }

        /*  modify map content  */
        pthread_mutex_lock(&map_mutex);

        for (i = 0; i < COLUMN - 1; ++i)
            map[tid + 1][i] = ' ';
        for (i = 0; i < LOG_LENGTH; i++) {
            j = (init_pos + i) % (COLUMN - 1);
            map[tid + 1][j] = '=';
        }
        pthread_cond_signal(&log_complete);
        pthread_mutex_unlock(&map_mutex);
    }
    pthread_exit(NULL);
}

void *frog_move(void *threads_p) {
    while (game_status == NORMAL) {
        usleep(REFRESH_TIME);
        pthread_mutex_lock(&map_mutex);
        pthread_cond_wait(&log_complete, &map_mutex);
        if (frog.on_log) {  // frog is on a log
            if (direction_is_left(frog.x)) frog.y--;  // left
            else frog.y++;  // right
        }

        /*  Check keyboard hits, to change frog's position or quit the game. */
        if (kbhit()) {
            char dir = getchar();
            if (dir == 'w' || dir == 'W') {  // up
                frog.x--;
                move_if_on_log(&frog);
            }
            if (dir == 's' || dir == 'S') {  // down
                if (frog.x < ROW) frog.x++;
                move_if_on_log(&frog);
            }
            if (dir == 'a' || dir == 'A') frog.y--;  // left
            if (dir == 'd' || dir == 'D') frog.y++;  // right
            if (dir == 'q' || dir == 'Q') game_status = QUIT;  // quit
        }

        /*  Check game's status  */
        if (map[frog.x][frog.y] == ' ' || frog.y < 0 || frog.y > COLUMN - 2) {  // frog dies
            game_status = LOSE;
            pthread_mutex_unlock(&map_mutex);
            break;
        }
        if (frog.x == 0) game_status = WIN;  // frog reaches the upper bank
        for (int j = 0; j < COLUMN - 1; ++j)
            map[ROW][j] = map[0][j] = '|';  // lower bank
        map[frog.x][frog.y] = '0';

        /*  Print the map into screen   */
        for (int i = 0; i <= ROW; ++i) {
            puts(map[i]);
        }
        usleep(REFRESH_TIME / 3);
        printf("\033[0;0H\033[2J");  // refresh screen
        pthread_mutex_unlock(&map_mutex);
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {

    /*  Initialize the river map and frog's starting position   */
    memset(map, 0, sizeof(map));
    printf("\033[?25l");  // hide cursor
    int i, j;
    const int NUM_THREAD = NUM_LOG + 1;
    for (i = 1; i < ROW; ++i) {  // river
        for (j = 0; j < COLUMN - 1; ++j)
            map[i][j] = ' ';
    }

    for (j = 0; j < COLUMN - 1; ++j)
        map[ROW][j] = map[0][j] = '|';  // lower bank

    for (j = 0; j < COLUMN - 1; ++j)
        map[0][j] = map[0][j] = '|';  // upper bank

    frog = Node(ROW, (COLUMN - 1) / 2);
    frog.on_log = false;

    /*  Create pthreads for wood move and frog control.  */
    pthread_t threads[NUM_THREAD];
    pthread_mutex_init(&map_mutex, NULL);
    pthread_cond_init(&log_complete, NULL);

    int rc;
    for (i = 0; i < NUM_LOG; i++) {
        rc = pthread_create(&threads[i], NULL, logs_move, (void *) i);
        usleep(10);  // ensure every log gets different random seed
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d", rc);
            exit(1);
        }
    }
    rc = pthread_create(&threads[i], NULL, frog_move, (void *) threads);
    if (rc) {
        printf("ERROR: return code from pthread_create() is %d", rc);
        exit(1);
    }

    /*  Display the output for user: win, lose or quit.  */

    for (int i = 0; i < NUM_THREAD; i++) {
        pthread_join(threads[i], NULL);
    }
    switch (game_status) {
        case QUIT:
            printf("You exit the game.\n");
            break;
        case WIN:
            printf("You win the game!!\n");
            break;
        case LOSE:
            printf("You lose the game!!\n");
            break;
        default:
            break;
    }

    pthread_mutex_destroy(&map_mutex);
    pthread_cond_destroy(&log_complete);
    pthread_exit(NULL);
    return 0;
}
