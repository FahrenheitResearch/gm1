"""One --follow writer per run folder (cm1out.follow_lock): the lock itself, and the quick-look follower."""

from __future__ import annotations

import threading
import time

import pytest

from cm1core.cm1out.follow_lock import FollowerStopped, follow_lock, lock_path


def test_a_second_holder_waits_until_the_first_lets_go(tmp_path):
    order: list[str] = []
    said: list[str] = []
    with follow_lock(tmp_path, "cm1out") as held:
        assert held and lock_path(tmp_path, "cm1out").is_file()

        def second() -> None:
            with follow_lock(tmp_path, "cm1out", log=said.append, poll_s=0.05):
                order.append("second")

        thread = threading.Thread(target=second)
        thread.start()
        time.sleep(0.4)
        assert thread.is_alive() and order == [] and len(said) == 1 and "waiting" in said[0]
        order.append("first done")
    thread.join(timeout=10)
    assert order == ["first done", "second"]


def test_a_stop_while_waiting_raises_and_other_names_do_not_wait(tmp_path):
    with follow_lock(tmp_path, "quicklook"):
        with pytest.raises(FollowerStopped, match="still writing"):
            with follow_lock(tmp_path, "quicklook", should_stop=lambda: True, poll_s=0.01):
                pytest.fail("the lock is held")
        with follow_lock(tmp_path, "cm1out") as held:  # cm1out and the quick-looks each have their own
            assert held


def test_a_folder_that_cannot_hold_the_lock_runs_unlocked(tmp_path):
    with follow_lock(tmp_path / "missing", "cm1out") as held:
        assert held is False


def test_the_quicklook_follower_takes_the_lock(tmp_path):
    from cm1core.quicklook.render import QuicklookError, render

    with follow_lock(tmp_path, "quicklook"):
        with pytest.raises(QuicklookError, match="still writing"):
            render(tmp_path, follow=True, should_stop=lambda: True)
