from unittest.mock import Mock
from pathlib import Path
from pretrain.main import save_checkpoint


def test_save_checkpoint():
    mocked_fabric = Mock()
    mocked_fabric.save = Mock()

    mocked_state = {
        'step_count': 10,  # divisible by `save_step_interval`
        'iter_num': 999,
    }

    out_dir = Path('foo')

    save_checkpoint(
        fabric=mocked_fabric,
        is_accumulating=False,
        state=mocked_state,
        save_step_interval=5,
        out_dir=out_dir,
    )

    mocked_fabric.save.assert_called_once_with(
        out_dir / f'step-{mocked_state["step_count"]:08d}/lit_model.pth', mocked_state
    )


def test_save_checkpoint_should_not_save_due_not_arrived_at_save_step():
    mocked_fabric = Mock()
    mocked_fabric.save = Mock()

    mocked_state = {
        'step_count': 9,  # NOT divisible by `save_step_interval`
        'iter_num': 999,
    }

    out_dir = Path('foo')

    save_checkpoint(
        fabric=mocked_fabric,
        is_accumulating=False,
        state=mocked_state,
        save_step_interval=5,
        out_dir=out_dir,
    )

    mocked_fabric.save.assert_not_called()


def test_save_checkpoint_should_not_save_due_still_accumulating():
    mocked_fabric = Mock()
    mocked_fabric.save = Mock()

    mocked_state = {
        'step_count': 10,  # divisible by `save_step_interval`
        'iter_num': 999,
    }

    out_dir = Path('foo')

    save_checkpoint(
        fabric=mocked_fabric,
        is_accumulating=True,
        state=mocked_state,
        save_step_interval=5,
        out_dir=out_dir,
    )

    mocked_fabric.save.assert_not_called()


def test_save_checkpoint_should_save_if_is_last_iteration():
    mocked_fabric = Mock()
    mocked_fabric.save = Mock()

    mocked_state = {
        'step_count': 10,  # divisible by `save_step_interval`
        'iter_num': 999,
    }

    out_dir = Path('foo')

    save_checkpoint(
        fabric=mocked_fabric,
        is_accumulating=True,
        state=mocked_state,
        save_step_interval=5,
        out_dir=out_dir,
    )

    mocked_fabric.save.assert_not_called()
