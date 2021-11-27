from paddle import inference
import time
import numpy as np


def load_predictor(
        model_file_path,
        params_file_path,
        enable_mkldnn=True, ):
    config = inference.Config(model_file_path, params_file_path)
    config.disable_gpu()
    if enable_mkldnn:
        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(10)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    return predictor


def test_time(model_prefix, batch_size=1):
    predictor = load_predictor(model_prefix + ".pdmodel",
                               model_prefix + ".pdiparams")

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    test_num = 5000
    warmup_time = 100
    test_time = 0.0
    for i in range(0, test_num + warmup_time):
        inputs = np.random.rand(batch_size, 768).astype(np.float32)
        start_time = time.time()
        input_tensor.copy_from_cpu(inputs)

        predictor.run()

        output = output_tensor.copy_to_cpu()
        output = output.flatten()
        if i >= warmup_time:
            test_time += time.time() - start_time
        time.sleep(0.001)

    avg = test_time / test_num * 1000  # ms
    print(f"model: {model_prefix}, bs: {batch_size}, avg time: {avg}")
    return avg


def main():
    test_time("base", 1)
    time.sleep(2)
    test_time("feat", 1)


if __name__ == "__main__":
    main()
