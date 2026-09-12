%%writefile train_pendulum.cpp

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include <iostream>
#include <fstream>

namespace rlt = rl_tools;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DEVICE_FACTORY<rlt::devices::DefaultCPUSpecification>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;

using ENVIRONMENT_PARAMETERS = rlt::rl::environments::pendulum::DefaultParameters<T>;
using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;

struct TD3_PARAMETERS : rlt::rl::algorithms::td3::DefaultParameters<TYPE_POLICY, TI> {
    static constexpr TI CRITIC_BATCH_SIZE = 100;
    static constexpr TI ACTOR_BATCH_SIZE = 100;
};

constexpr TI STEP_LIMIT = 10000;
constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
constexpr TI EPISODE_STEP_LIMIT = 200;
constexpr TI ACTOR_NUM_LAYERS = 3;
constexpr TI ACTOR_HIDDEN_DIM = 64;
constexpr TI CRITIC_NUM_LAYERS = 3;
constexpr TI CRITIC_HIDDEN_DIM = 64;
constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;
constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;
constexpr auto ACTOR_ACTIVATION_FUNCTION_OUTPUT = rlt::nn::activation_functions::TANH;
constexpr auto CRITIC_ACTIVATION_FUNCTION_OUTPUT = rlt::nn::activation_functions::IDENTITY;

using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TD3_PARAMETERS::ACTOR_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
using CRITIC_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TD3_PARAMETERS::CRITIC_BATCH_SIZE, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM>;
using ACTOR_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, ACTOR_NUM_LAYERS, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION, ACTOR_ACTIVATION_FUNCTION_OUTPUT>;
using CRITIC_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, CRITIC_NUM_LAYERS, CRITIC_HIDDEN_DIM, CRITIC_ACTIVATION_FUNCTION, CRITIC_ACTIVATION_FUNCTION_OUTPUT>;
using OPTIMIZER_SPEC = typename rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI>;
using PARAMETER_TYPE = rlt::nn::parameters::Adam;
using CAPABILITY_ACTOR = rl_tools::nn::capability::Gradient<PARAMETER_TYPE>;
using CAPABILITY_CRITIC = rl_tools::nn::capability::Gradient<PARAMETER_TYPE>;
using CAPABILITY_TARGET = rl_tools::nn::capability::Forward<>;

using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using ACTOR_TYPE = rlt::nn_models::mlp::Build<ACTOR_CONFIG, CAPABILITY_ACTOR, ACTOR_INPUT_SHAPE>;
using ACTOR_TARGET_TYPE = rl_tools::nn_models::mlp::NeuralNetwork<ACTOR_CONFIG, CAPABILITY_TARGET, ACTOR_INPUT_SHAPE>;
using CRITIC_TYPE = rlt::nn_models::mlp::Build<CRITIC_CONFIG, CAPABILITY_CRITIC, CRITIC_INPUT_SHAPE>;
using CRITIC_TARGET_TYPE = rl_tools::nn_models::mlp::NeuralNetwork<CRITIC_CONFIG, CAPABILITY_TARGET, CRITIC_INPUT_SHAPE>;

using TD3_SPEC = rlt::rl::algorithms::td3::Specification<TYPE_POLICY, DEVICE::index_t, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, TD3_PARAMETERS>;
using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

struct OFF_POLICY_RUNNER_PARAMETERS : rlt::rl::components::off_policy_runner::ParametersDefault<T, TI> {};
using EXPLORATION_POLICY_SPEC = rlt::nn_models::random_uniform::Specification<TYPE_POLICY, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, rlt::nn_models::random_uniform::Range::MINUS_ONE_TO_ONE>;
using EXPLORATION_POLICY = rlt::nn_models::RandomUniform<EXPLORATION_POLICY_SPEC>;
using POLICIES = rl_tools::utils::Tuple<TI, EXPLORATION_POLICY, ACTOR_TYPE>;

using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<TYPE_POLICY, TI, ENVIRONMENT, POLICIES, OFF_POLICY_RUNNER_PARAMETERS>;
using OFF_POLICY_RUNNER_TYPE = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;

int main() {
    DEVICE device;
    RNG rng;
    TI seed = 1;
    rlt::init(device, rng, seed);

    OPTIMIZER actor_optimizer, critic_optimizers[2];
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, critic_optimizers[0]);
    rlt::malloc(device, critic_optimizers[1]);

    ACTOR_CRITIC_TYPE actor_critic;
    rlt::malloc(device, actor_critic);
    rlt::init(device, actor_critic, rng);

    OFF_POLICY_RUNNER_TYPE runner;
    rlt::malloc(device, runner);
    ENVIRONMENT env;
    auto& envs = rlt::get_envs(device, runner);
    rlt::malloc(device, envs[0], env);
    rlt::init(device, runner, rng);

    // 学習ループ
    for (TI step = 0; step < STEP_LIMIT; ++step) {
        rlt::step(device, runner, actor_critic, rng);
        if (step > TD3_PARAMETERS::ACTOR_BATCH_SIZE) {
            rlt::train_critic(device, actor_critic, critic_optimizers, runner.replay_buffer, rng);
            if (step % 2 == 0) {
                rlt::train_actor(device, actor_critic, actor_optimizer, runner.replay_buffer, rng);
                rlt::update_critic_targets(device, actor_critic);
                rlt::update_actor_target(device, actor_critic);
            }
        }
        if (step % 1000 == 0) {
            T mean_return = rlt::evaluate(device, env, envs[0], actor_critic.actor, rng, EPISODE_STEP_LIMIT);
            std::cout << "Step: " << step << "/" << STEP_LIMIT << " Mean return: " << mean_return << std::endl;
        }
    }

    // チェックポイントをファイルに出力
    std::ofstream checkpoint_file("checkpoint.h");
    checkpoint_file << rlt::get_model_weights_inlined(device, actor_critic.actor, "policy");
    checkpoint_file.close();

    std::cout << "学習完了。checkpoint.h を出力しました。" << std::endl;

    rlt::free(device, actor_critic);
    rlt::free(device, runner);
    rlt::free(device, actor_optimizer);
    rlt::free(device, critic_optimizers[0]);
    rlt::free(device, critic_optimizers[1]);

    return 0;
}