仔细阅读./scripts/dev/GOAL.md中的任务，并按任务说明执行

按GOAL.md的任务目的，继续后面的任务，先跑完整的基线

切换回8.5.0-skip分支，按GOAL.md的任务目的开始进行skip功能的代码开发

按ascend_fused_infer_attention_score_decode_cycle.sh脚本执行8.5.0-skip分支的编译、测试，可能会报错，如果报错，根据编译日志、测试日志修正代码，编译时间很久大概要20分钟。基线分支的编译、测试已经做了，相关日志在.codex-remote-logs/20260228-223518，不用重复编译测试基线分支。

注意是在远程服务器的tf docker容器中进行编译测试，ascend_fused_infer_attention_score_decode_cycle.sh中应该有说明

