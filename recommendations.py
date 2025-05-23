# recommendations.py
from typing import List

__all__ = ['generate_recommendations']

def generate_recommendations(commit: dict,
                             risk_proba: float,
                             repo_stats: dict,
                             feature_importances: dict) -> List[str]:
    recommendations: List[str] = []

    # Оценка общего риска
    if risk_proba > 0.8:
        recommendations.append(
            "⚠️ Очень высокий риск: обязательно провести углублённое код-ревью и расширенное тестирование."
        )
        # Показываем feature_importances только при высоком риске
        if feature_importances:
            top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
            important_list = ', '.join(f"{feat}" for feat, _ in top_features)
            recommendations.append(
                f"📌 Наибольшее влияние на риск оказали признаки: {important_list}."
            )
    elif risk_proba > 0.5:
        recommendations.append(
            "🔍 Повышенный риск: обратите внимание на качество изменений и добавьте тесты."
        )

    # Длина сообщения
    msg_len = commit.get('message_length', 0)
    if msg_len < 15:
        recommendations.append("✏️ Сообщение слишком короткое: дайте подробное описание изменений.")
    elif msg_len > 200:
        recommendations.append("📝 Очень длинное сообщение: разделите описание на ключевые пункты или используйте более лаконичные формулировки.")

    # Наличие ключевых слов (например, багфикса)
    if commit.get('has_bug_keyword', 0):
        recommendations.append("🐞 Выявлен багфикс: убедитесь в наличии регрессионных тестов и обновлении документации.")

    # Объём изменений
    lines_added = commit.get('lines_added', 0)
    lines_deleted = commit.get('lines_deleted', 0)
    total = lines_added + lines_deleted
    stats_total = repo_stats.get('total_changes', {})
    mean_total = stats_total.get('mean')
    std_total = stats_total.get('std')
    if mean_total is not None and std_total is not None and total > mean_total + 2 * std_total:
        recommendations.append(
            f"📊 Объём изменений ({total}) значительно превышает среднее ({mean_total:.1f}). "
            "Разбейте коммит на более мелкие логические части."
        )

    # Число затронутых файлов
    files_changed = commit.get('files_changed', 0)
    q95_files = repo_stats.get('files_changed', {}).get('quantile_95')
    if q95_files is not None and files_changed > q95_files:
        recommendations.append(
            f"📂 Затронуто слишком много файлов ({files_changed} > 95% квантиль). Проверьте целостность изменений."
        )

    # Сложность изменений
    complexity = commit.get('complexity_score', 0)
    q90_complex = repo_stats.get('complexity_score', {}).get('quantile_90')
    if q90_complex is not None and complexity > q90_complex:
        recommendations.append(
            f"🧩 Высокая сложность ({complexity} > 90% квантиль). "
            "Рассмотрите рефакторинг и дополнительное покрытие тестами."
        )

    # История файлов
    avg_hist = commit.get('avg_file_history', 0)
    stats_hist = repo_stats.get('avg_file_history', {})
    mean_hist = stats_hist.get('mean')
    std_hist = stats_hist.get('std')
    if mean_hist is not None and std_hist is not None and avg_hist > mean_hist + 2 * std_hist:
        recommendations.append(
            f"📈 Файлы часто меняются ({avg_hist:.1f} > {mean_hist:.1f} + 2σ). "
            "Возможно, стоит разделить функциональность."
        )

    # Интервал между коммитами
    interval = commit.get('minutes_since_previous_commit')
    median_int = repo_stats.get('commit_interval', {}).get('median')
    if interval is not None and median_int is not None:
        if interval < 5:
            recommendations.append("⏱ Очень быстрый коммит (<5 минут): убедитесь, что изменения завершены и протестированы.")
        elif interval > 2 * median_int:
            recommendations.append(
                f"⏳ Промежуток {interval:.0f} мин более чем в 2 раза дольше медианы "
                f"({median_int:.0f} мин): проверьте актуальность ветки перед слиянием."
            )

    # Поведение автора
    author = commit.get('author_name', 'Unknown')
    median_lines_author = repo_stats.get('author_stats', {}).get(author, {}).get('median_lines_added')
    if median_lines_author is not None and lines_added > 2 * median_lines_author:
        recommendations.append(
            f"👤 Автор {author} внёс {lines_added} строк, что более чем в 2 раза превышает "
            f"его медианные {median_lines_author} строк: дополнительная проверка кода."
        )

    # Если рекомендаций не было
    if not recommendations:
        recommendations.append("✅ Явных аномалий не обнаружено. Рекомендуется стандартное код-ревью и покрытие тестами.")

    return recommendations
