"""Стани FSM."""
from aiogram.fsm.state import State, StatesGroup


class RegistrationStates(StatesGroup):
    """Очікуємо справжнє ім'я користувача перед збереженням його першого документа."""

    waiting_for_name = State()
