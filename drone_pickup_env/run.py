from safe_rl import cpo
from gridworld_irl import GridWorld
from safe_rl.utils.run_utils import setup_logger_kwargs

def train(cl):
    seed = 10
    exp_name = "CPO_Gridworld_CL{}".format(cl)
    logger_kwargs = setup_logger_kwargs(exp_name, seed, "./data/")

    total_time_steps = 1000000
    steps_per_epoch = 2000
    epochs = int(total_time_steps / steps_per_epoch)
    cpo(
        env_fn = lambda : GridWorld(cl),
        ac_kwargs = dict(hidden_sizes=(256,256)),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        save_freq=100,
        cost_lim=cl,
        seed=seed,
        max_ep_len=200,
        logger_kwargs=logger_kwargs
        )

if __name__=="__main__":
    train(50)

