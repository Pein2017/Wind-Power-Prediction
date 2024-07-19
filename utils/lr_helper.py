from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.learning_rate
            if epoch < 3
            else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "PEMS":
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print(f"\nUpdating learning rate to {lr}\n")
