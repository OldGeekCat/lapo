"""lr train CLI — 薄壳，调 services 层。

子命令:
  train      跑训练: lr train --config exp.yaml
  registry   管理 policy/strategy/trait 注册条目
  env        环境检测 + 兼容性诊断
"""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="train",
    help="配置驱动的训练框架：声明 policy，其余全自动。",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def train(
    config: str = typer.Option(..., "--config", help="实验 YAML 路径"),
) -> None:
    """跑一次训练：lr train --config exp.yaml"""
    from lapo.train.config import load_config
    from lapo.train.services.training import load_registry_with_builtins, run_training
    from lapo.train.env_check import collect_env
    from lapo.paths import outputs_dir

    cfg = load_config(config)
    registry = load_registry_with_builtins()
    console.print(f"[bold]policy:[/bold] {cfg.policy_name}  "
                  f"[bold]strategy:[/bold] {cfg.strategy_name or '(auto)'}")
    run_dir = run_training(
        cfg, registry=registry, outputs_root=outputs_dir(),
        env_info=collect_env().to_dict(),
    )
    console.print(f"[green]✓ 训练完成。产物: {run_dir}[/green]")


# ---- registry 子命令 ----
registry_app = typer.Typer(help="管理 policy/strategy/trait 注册条目。",
                           no_args_is_help=True)
app.add_typer(registry_app, name="registry")


@registry_app.command("add-policy")
def add_policy(
    name: str = typer.Argument(..., help="policy 短名"),
    config_class: str = typer.Option(..., "--config-class", help="完整 Python 路径"),
    policy_class: Optional[str] = typer.Option(None, "--policy-class"),
    trait: list[str] = typer.Option([], "--trait", help="可重复"),
    default_strategy: Optional[str] = typer.Option(None, "--default-strategy"),
) -> None:
    """注册一个 policy。"""
    from lapo.train.registry import PolicyEntry, Registry
    from lapo.train.services.registry_store import RegistryStore
    Registry(RegistryStore()).register_policy(PolicyEntry(
        name=name, config_cls=config_class, policy_cls=policy_class,
        traits=set(trait), default_strategy=default_strategy,
    ))
    console.print(f"[green]✓ 已注册 policy '{name}'[/green]")


@registry_app.command("add-strategy")
def add_strategy(
    name: str = typer.Argument(..., help="strategy 短名"),
    class_path: str = typer.Option(..., "--class-path", help="完整 Python 路径"),
    requires_trait: list[str] = typer.Option([], "--requires-trait", help="可重复"),
) -> None:
    """注册一个 strategy。"""
    from lapo.train.registry import StrategyEntry, Registry
    from lapo.train.services.registry_store import RegistryStore
    Registry(RegistryStore()).register_strategy(StrategyEntry(
        name=name, cls_path=class_path, required_traits=set(requires_trait),
    ))
    console.print(f"[green]✓ 已注册 strategy '{name}'[/green]")


@registry_app.command("add-trait")
def add_trait(name: str = typer.Argument(...)) -> None:
    """登记一个新 trait（扩展词表）。"""
    from lapo.train.registry import Registry
    from lapo.train.services.registry_store import RegistryStore
    Registry(RegistryStore()).register_trait(name)
    console.print(f"[green]✓ 已登记 trait '{name}'[/green]")


@registry_app.command("list")
def list_entries(
    type: Optional[str] = typer.Option(None, "--type",
                                       help="policy|strategy|trait；不指定则全列"),
) -> None:
    """列出注册条目。"""
    from lapo.train.services.training import load_registry_with_builtins
    reg = load_registry_with_builtins()
    if type in (None, "policy"):
        t = Table("policy", "traits", "default_strategy", title="policies")
        for name in sorted(reg.list_policies()):
            e = reg.get_policy(name)
            t.add_row(name, ",".join(sorted(e.traits)) or "-", e.default_strategy or "-")
        console.print(t)
    if type in (None, "strategy"):
        t = Table("strategy", "required_traits", title="strategies")
        for name in sorted(reg.list_strategies()):
            e = reg.get_strategy(name)
            t.add_row(name, ",".join(sorted(e.required_traits)) or "-")
        console.print(t)
    if type in (None, "trait"):
        console.print("traits: " + ", ".join(reg.list_traits()))


@registry_app.command("remove")
def remove(
    name: str = typer.Argument(...),
    type: str = typer.Option(..., "--type", help="policy|strategy|trait"),
) -> None:
    """移除注册条目。"""
    from lapo.train.registry import Registry
    from lapo.train.services.registry_store import RegistryStore
    reg = Registry(RegistryStore())
    if type == "policy":
        reg.remove_policy(name)
    elif type == "strategy":
        reg.remove_strategy(name)
    elif type == "trait":
        if not reg.remove_trait(name):
            console.print(f"[yellow]trait '{name}' 不存在或为内置（内置 trait 不可删）[/yellow]")
            raise typer.Exit(code=1)
    else:
        raise typer.BadParameter("type 必须是 policy|strategy|trait")
    console.print(f"[green]✓ 已移除 {type} '{name}'[/green]")


# ---- env 命令 ----
@app.command("env")
def env(
    json_out: bool = typer.Option(False, "--json", help="JSON 输出"),
) -> None:
    """检测环境 + 兼容性诊断。"""
    import json as _json
    from lapo.train.env_check import collect_env, check_env
    info = collect_env()
    results = check_env(info)
    if json_out:
        print(_json.dumps({
            "env": info.to_dict(),
            "checks": [{"name": r.name, "status": r.status, "message": r.message}
                       for r in results],
        }, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]python:[/bold] {info.python}")
    console.print(f"[bold]torch:[/bold] {info.torch}  "
                  f"[bold]lerobot:[/bold] {info.lerobot}  "
                  f"[bold]transformers:[/bold] {info.transformers}")
    if info.gpu_name:
        console.print(f"[bold]GPU:[/bold] {info.gpu_name} "
                      f"(CUDA {info.cuda}, cuDNN {info.cudnn})")
    console.print(f"[bold]git:[/bold] {info.git_commit}")
    console.print("\n[bold]兼容性:[/bold]")
    for r in results:
        color = {"ok": "green", "warn": "yellow", "fail": "red"}[r.status]
        console.print(f"  [{color}]{r.status}[/{color}] {r.name}: {r.message}")


if __name__ == "__main__":
    app()
