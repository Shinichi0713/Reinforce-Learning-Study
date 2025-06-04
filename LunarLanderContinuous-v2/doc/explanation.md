## 環境について
Box2D

Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
LunarLander-v2 defines "solving" as getting average reward of 200 over 100 consecutive trials.=
by @olegklimov


https://elegantrl.readthedocs.io/en/latest/tutorial/LunarLanderContinuous-v2.html


how to use
```
pip install gym[box2d]
```

なんか自己証明書の検証に失敗する・・・
```
pip install gym[box2d] --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

