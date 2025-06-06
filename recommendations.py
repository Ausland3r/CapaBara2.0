import re
from typing import List, Optional


class RecommendationGenerator:
    def __init__(self):
        pass

    def generate_repo_level_recommendations(self,
                                            repo_files: Optional[List[str]] = None,
                                            readme_content: Optional[str] = None) -> List[str]:
        recommendations = []
        readme_present = False
        if repo_files:
            readme_present = any('readme' in f.lower() for f in repo_files)

        if not readme_present:
            recommendations.append(
                "ℹ️ В репозитории отсутствует README. Рекомендуется добавить README с описанием проекта."
            )

        return recommendations

    def generate_recommendations(self,
                                 commit: dict,
                                 risk_proba: float,
                                 repo_stats: dict,
                                 feature_importances: dict,
                                 repo_files: Optional[List[str]] = None,
                                 readme_content: Optional[str] = None) -> List[str]:
        recommendations: List[str] = []

        message = commit.get('message', '')
        author = commit.get('author_name', 'Unknown')
        lines_added = commit.get('lines_added', 0)
        files_changed = commit.get('files_changed', 0)
        complexity = commit.get('complexity_score', 0)
        interval = commit.get('minutes_since_previous_commit')
        has_bug_keyword = commit.get('has_bug_keyword', 0)

        if 'merge' in message.lower():
            recommendations.append(
                "⚠️ Merge-коммиты лучше делать с использованием стандартных сообщений (например, squash или fast-forward) для удобства чтения истории."
            )


        if re.search(r'[^\x00-\x7F]', message):
            recommendations.append(
                "⚠️ В сообщении обнаружены нестандартные символы. Желательно использовать только ASCII.")

        if lines_added + commit.get('lines_deleted', 0) > 500 or files_changed > 20:
            recommendations.append(
                "⚠️ Коммит слишком большой — рекомендуется разбить изменения на более мелкие логические части."
            )

        if len(message.strip()) < 10:
            recommendations.append("⚠️ Сообщение слишком короткое, оно не даёт понимания изменений.")

        if commit.get('author_name') != commit.get('committer_name', commit.get('author_name')):
            recommendations.append("⚠️ Автор и коммиттер не совпадают — проверьте правильность указания данных.")

        if commit.get('tests_added', 0) == 0 and has_bug_keyword:
            recommendations.append(
                "⚠️ Для исправления бага не добавлены тесты. Добавьте тесты для проверки исправления.")

        if any(f.lower().endswith(('.zip', '.pickle', '.exe', '.dll', '.bin')) for f in commit.get('files', [])):
            recommendations.append("⚠️ В коммите есть бинарные файлы. Лучше хранить бинарники вне репозитория.")

        if interval is not None and interval < 5:
            recommendations.append(
                "⚠️ Коммит сделан слишком быстро (меньше 5 минут). Проверьте полноту и качество изменений.")

        if risk_proba > 0.8:
            recommendations.append(
                "⚠️ Высокий риск: рекомендуется провести детальный код-ревью и расширенное тестирование."
            )
            if feature_importances:
                top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
                important_list = ', '.join(f"{feat}" for feat, _ in top_features)
                recommendations.append(
                    f"📌 Основные факторы риска: {important_list}."
                )
        elif risk_proba > 0.5:
            recommendations.append(
                "🔍 Средний риск: обратите внимание на качество изменений и наличие тестов."
            )

        msg_len = commit.get('message_length', 0)
        if msg_len < 15:
            recommendations.append("✏️ Сообщение очень короткое, пожалуйста, опишите изменения подробнее.")
        elif msg_len > 200:
            recommendations.append("📝 Сообщение слишком длинное, постарайтесь сделать его короче и структурированнее.")

        if has_bug_keyword:
            recommendations.append(
                "🐞 Найден багфикс: проверьте наличие регрессионных тестов и обновление документации.")

        stats_total = repo_stats.get('total_changes', {})
        mean_total = stats_total.get('mean')
        std_total = stats_total.get('std')
        total = lines_added + commit.get('lines_deleted', 0)
        if mean_total is not None and std_total is not None and total > mean_total + 2 * std_total:
            recommendations.append(
                f"📊 Объём изменений ({total}) значительно больше среднего ({mean_total:.1f}). Разбейте коммит на части."
            )

        q95_files = repo_stats.get('files_changed', {}).get('quantile_95')
        if q95_files is not None and files_changed > q95_files:
            recommendations.append(
                f"📂 Изменено слишком много файлов ({files_changed} > 95-й процентиль). Проверьте корректность изменений."
            )

        q90_complex = repo_stats.get('complexity_score', {}).get('quantile_90')
        if q90_complex is not None and complexity > q90_complex:
            recommendations.append(
                f"🧩 Изменения слишком сложные ({complexity} > 90-й процентиль). Рассмотрите рефакторинг и дополнительные тесты."
            )

        stats_hist = repo_stats.get('avg_file_history', {})
        mean_hist = stats_hist.get('mean')
        std_hist = stats_hist.get('std')
        avg_hist = commit.get('avg_file_history', 0)
        if mean_hist is not None and std_hist is not None and avg_hist > mean_hist + 2 * std_hist:
            recommendations.append(
                f"📈 Файлы меняются слишком часто ({avg_hist:.1f} > {mean_hist:.1f} + 2σ). Возможно, стоит разделить функциональность."
            )

        median_int = repo_stats.get('commit_interval', {}).get('median')
        if interval is not None and median_int is not None:
            if interval < 5:
                recommendations.append(
                    "⏱ Коммит сделан очень быстро (меньше 5 минут). Убедитесь, что изменения полностью завершены и протестированы."
                )
            elif interval > 2 * median_int:
                recommendations.append(
                    f"⏳ Интервал между коммитами {interval:.0f} мин — более чем в два раза больше медианы ({median_int:.0f} мин). Проверьте актуальность ветки."
                )

        median_lines_author = repo_stats.get('author_stats', {}).get(author, {}).get('median_lines_added')
        if median_lines_author is not None and lines_added > 2 * median_lines_author:
            recommendations.append(
                f"👤 Автор {author} внёс {lines_added} строк — это более чем в 2 раза выше его обычного объёма. Рекомендуется дополнительное ревью."
            )

        if not recommendations:
            recommendations.append(
                "✅ Явных проблем не обнаружено. Рекомендуется стандартное код-ревью и покрытие тестами.")

        return recommendations
