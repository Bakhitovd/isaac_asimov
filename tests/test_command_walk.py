"""Physical-command and evaluation regressions for continuous walking."""
import copy
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_rl.command_walk import CommandWalkEnv, DEFAULTS, assess_trace, segments_for
from mujoco_rl.command_walk_eval import progress_made, require_contract
from mujoco_rl.command_walk_train import create_model, save_bundle
from mujoco_rl.skill_env import SkillEnv
from mujoco_rl.skill_train import _stats_path

SOURCE = Path('mujoco_rl/checkpoints/nav/accepted.zip')


def good_trace(v=.2, w=0.):
    return [dict(segment=0, age=(i+1)*.02, command_v=v, command_w=w, requested_v=v,
                 requested_w=w, vx=v, yaw_rate=w, heading_error=0., lateral=0.,
                 forward=v*(i+1)*.02, displacement=0. if v==0 else v*(i+1)*.02,
                 rotation=w*(i+1)*.02, speed=v, fallen=False, self_contacts=0, slip=0.)
            for i in range(400)]


def test_straight_and_intentional_turns_pass():
    for v,w in ((.2,0.),(.2,.3),(.2,-.3),(0.,.3),(0.,-.3),(0.,0.)):
        assert assess_trace(good_trace(v,w))['success']


def test_circle_cannot_pass_by_returning_to_start():
    rows=good_trace()
    for i,r in enumerate(rows):
        angle=2*math.pi*(i+1)/len(rows)
        r.update(heading_error=-angle, lateral=math.cos(angle)-1, forward=math.sin(angle))
    result=assess_trace(rows)
    assert not result['success']
    assert 'segment0/heading_max' in result['failures']
    assert 'segment0/lateral_max' in result['failures']
    assert 'segment0/progress_error' in result['failures']


def test_temporary_drift_and_extra_revolution_fail():
    rows=good_trace();rows[30]['lateral']=.4
    assert 'segment0/lateral_max' in assess_trace(rows)['failures']
    rows=good_trace(0.,.3)
    for row in rows:row['heading_error']=2*math.pi
    assert 'segment0/heading_max' in assess_trace(rows)['failures']


def test_wrong_turn_and_late_stop_fail():
    rows=good_trace(0.,.3);rows[-1]['rotation']=-1.
    assert 'segment0/wrong_turn_direction' in assess_trace(rows)['failures']
    rows=good_trace(0.,0.);rows[110]['speed']=.1
    assert 'segment0/stop_speed' in assess_trace(rows)['failures']


def test_missing_segment_time_fall_and_collision_fail():
    assert not assess_trace(good_trace()[:30],False)['success']
    for key in ('fallen','self_contacts'):
        rows=good_trace();rows[0][key]=True
        assert not assess_trace(rows)['success']


def test_reset_diversity_reproducibility_and_frame_rotation():
    env=CommandWalkEnv(family='forward')
    try:
        a,_=env.reset(seed=1);pose=env.data.qpos.copy()
        b,_=env.reset(seed=2)
        assert not np.array_equal(a,b)
        c,_=env.reset(seed=1)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(pose,env.data.qpos)
        assert a.shape==(85,) and np.all(a[81:83]==0)
        # Rotating the entire initial configuration and heading reference is invariant.
        obs=env._observe(False)
        quat=np.zeros(4);mujoco.mju_axisAngle2Quat(quat,np.array([0.,0.,1.]),1.2)
        rotated=np.zeros(4);mujoco.mju_mulQuat(rotated,quat,env.data.qpos[3:7])
        env.data.qpos[3:7]=rotated;env.reference_heading+=1.2
        mujoco.mj_forward(env.model,env.data)
        np.testing.assert_allclose(obs,env._observe(False),atol=2e-6)
    finally:env.close()


def test_commands_signs_timeout_and_continuous_state():
    now=[0.]
    env=CommandWalkEnv(external=True,clock=lambda:now[0])
    try:
        env.reset(seed=4)
        for yaw in (.6,-.6):
            pose=env.data.qpos.copy();buffer=[x.copy() for x in env.sensor_buffer]
            env.set_command(.3,yaw)
            np.testing.assert_array_equal(pose,env.data.qpos)
            np.testing.assert_array_equal(buffer,list(env.sensor_buffer))
            assert env.requested[1]==yaw
        env.set_command(.3,.6)
        for _ in range(20):env._prepare_command()
        assert env.command[0]>0 and env.command[1]>0
        now[0]=.6
        for _ in range(50):env._prepare_command()
        np.testing.assert_allclose(env.command,0.,atol=1e-10)
        env.set_command(100.,-100.)
        np.testing.assert_allclose(env.requested,[.3,-.6])
        with pytest.raises(ValueError):env.set_command(float('nan'),0.)
    finally:env.close()


def test_successful_speed_alone_does_not_count_as_progress():
    before={'stage':0,'minimum_pass_rate':0.,'mean_pass_rate':0.,'mean_violation':2.,
            'families':{'forward':{'mean_seconds':12.,'metrics':{'speed_error':.01}}}}
    after=copy.deepcopy(before);after['families']['forward']['metrics']['speed_error']=.001
    assert not progress_made(before,after)
    after['mean_violation']=1.7
    assert progress_made(before,after)


def test_warm_start_preserves_means_and_normalization_and_resume(tmp_path):
    torch.set_num_threads(1)
    config={**DEFAULTS,'workers':1,'rollout_steps':8,'batch_size':8,'epochs':1,'development_episodes':1}
    old=PPO.load(SOURCE,device='cpu')
    old_norm=VecNormalize.load(str(_stats_path(SOURCE)),DummyVecEnv([lambda:SkillEnv('nav')]))
    model,norm,resume,rng=create_model(SOURCE,44,config)
    try:
        obs=np.random.default_rng(12).normal(size=(10,83)).astype(np.float32)
        expanded=np.c_[obs,np.zeros((10,2))].astype(np.float32)
        with torch.no_grad():
            a=old.policy.get_distribution(torch.tensor(old_norm.normalize_obs(obs))).distribution.mean.numpy()
            b=model.policy.get_distribution(torch.tensor(norm.normalize_obs(expanded))).distribution.mean.numpy()
        np.testing.assert_allclose(a,b,atol=1e-6)
        mean=norm.obs_rms.mean.copy();norm.obs_rms.update(expanded)
        np.testing.assert_array_equal(mean,norm.obs_rms.mean)
        assert not resume and model.policy.mlp_extractor.policy_net[0].weight[:,83:].count_nonzero()==0
        model.command_walk_state['stage']=2
        path=save_bundle(model,norm,tmp_path,'initial')
        restored,rnorm,is_resume,rng2=create_model(path,45,config)
        try:
            assert is_resume and rng != rng2
            assert restored.command_walk_state['stage']==2
            np.testing.assert_array_equal(rnorm.obs_rms.mean,mean)
            require_contract(restored)
        finally:rnorm.close()
        with pytest.raises(ValueError):require_contract(old)
    finally:
        norm.close();old_norm.close()


def test_mixed_schedule_includes_both_pivots_and_stop():
    schedule=segments_for('mixed',.2,.3)
    assert (4.,0.,.3) in schedule and (4.,0.,-.3) in schedule
    assert schedule[-1]==(4.,0.,0.)


def test_pushes_during_motion_but_not_during_controlled_stop():
    env=CommandWalkEnv(stage=3,external=True)
    try:
        env.reset(seed=4)
        env.next_push=1
        _,_,_,_,info=env.step(np.zeros(23))
        assert not info['push_applied']
        env.set_command(.2,0.)
        env.next_push=env.step_count+1
        _,_,_,_,info=env.step(np.zeros(23))
        assert info['push_applied']
    finally:env.close()
