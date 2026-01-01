"""
Manual evaluation module for retrieval test results.

This module provides functionality to:
- Load retrieval test results from JSON
- Display questions and answers for manual evaluation
- Rate retrieved answers (1-5 scale)
- Add comments
- Save evaluations back to JSON
- Navigate between questions with fast access buttons
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gradio as gr


class ManualEvaluator:
    """Manages manual evaluation of retrieval test results."""
    
    def __init__(self, json_path: str = "retrieval_test_results.json"):
        self.json_path = Path(json_path)
        self.data: List[Dict] = []
        self.current_index = 0
        self.load_data()
    
    def load_data(self):
        """Load data from JSON file."""
        if self.json_path.exists():
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = []
            print(f"Warning: {self.json_path} not found. Starting with empty data.")
    
    def save_data(self):
        """Save data back to JSON file."""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        return "✅ Saved successfully!"
    
    def get_current_item(self) -> Optional[Dict]:
        """Get the current item being evaluated."""
        if 0 <= self.current_index < len(self.data):
            return self.data[self.current_index]
        return None
    
    def get_total_count(self) -> int:
        """Get total number of items."""
        return len(self.data)
    
    def get_evaluated_count(self) -> int:
        """Get count of items that have been evaluated (have rating)."""
        return sum(1 for item in self.data if item.get('manual_rating') is not None)
    
    def get_evaluated_indices(self) -> List[int]:
        """Get list of indices that have been evaluated."""
        return [i for i, item in enumerate(self.data) if item.get('manual_rating') is not None]
    
    def update_evaluation(self, rating: Optional[int], comment: str):
        """Update evaluation for current item."""
        if 0 <= self.current_index < len(self.data):
            if rating is not None:
                self.data[self.current_index]['manual_rating'] = rating
            if comment:
                self.data[self.current_index]['manual_comment'] = comment
            elif 'manual_comment' in self.data[self.current_index]:
                # Keep existing comment if new one is empty
                pass
            return True
        return False
    
    def navigate(self, direction: str) -> Tuple[Optional[Dict], str, int, int, List[int]]:
        """
        Navigate to next/previous item.
        
        Returns:
            Tuple of (current_item, status_message, current_index, total_count, evaluated_indices)
        """
        if direction == "next":
            if self.current_index < len(self.data) - 1:
                self.current_index += 1
            else:
                return self.get_current_item(), "⚠️ Already at last item", self.current_index, len(self.data), self.get_evaluated_indices()
        elif direction == "previous":
            if self.current_index > 0:
                self.current_index -= 1
            else:
                return self.get_current_item(), "⚠️ Already at first item", self.current_index, len(self.data), self.get_evaluated_indices()
        elif direction == "first":
            self.current_index = 0
        elif direction == "last":
            self.current_index = len(self.data) - 1
        
        item = self.get_current_item()
        status = f"📄 Item {self.current_index + 1} of {len(self.data)}"
        return item, status, self.current_index, len(self.data), self.get_evaluated_indices()
    
    def jump_to_index(self, index: int) -> Tuple[Optional[Dict], str, int, int, List[int]]:
        """Jump to a specific index."""
        if 0 <= index < len(self.data):
            self.current_index = index
            item = self.get_current_item()
            status = f"📄 Jumped to item {self.current_index + 1} of {len(self.data)}"
            return item, status, self.current_index, len(self.data), self.get_evaluated_indices()
        else:
            item = self.get_current_item()
            status = f"⚠️ Invalid index. Must be between 0 and {len(self.data) - 1}"
            return item, status, self.current_index, len(self.data), self.get_evaluated_indices()
    
    def format_item_display(self, item: Optional[Dict]) -> str:
        """Format item for display in HTML."""
        if item is None:
            return "<p>No item available</p>"
        
        html_parts = []
        html_parts.append("<div style='background: #2b2b2b; padding: 20px; border-radius: 10px; color: #e0e0e0;'>")
        
        # ID
        html_parts.append(f"<h3 style='color: #66b3ff; margin-top: 0;'>ID: {item.get('id', 'N/A')}</h3>")
        
        # Question
        html_parts.append("<div style='background: #3a3a3a; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #66b3ff;'>")
        html_parts.append("<h4 style='color: #ffffff; margin-top: 0;'>❓ Question:</h4>")
        html_parts.append(f"<p style='color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.1em; white-space: pre-wrap;'>{item.get('question', 'N/A')}</p>")
        html_parts.append("</div>")
        
        # Original Answer
        html_parts.append("<div style='background: #3a3a3a; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #66ff66;'>")
        html_parts.append("<h4 style='color: #ffffff; margin-top: 0;'>✅ Original Answer:</h4>")
        html_parts.append(f"<p style='color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.1em; white-space: pre-wrap;'>{item.get('original_answer', 'N/A')}</p>")
        html_parts.append("</div>")
        
        # Retrieved Answer
        html_parts.append("<div style='background: #3a3a3a; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ff9966;'>")
        html_parts.append("<h4 style='color: #ffffff; margin-top: 0;'>🔍 Retrieved Answer:</h4>")
        retrieved = item.get('retrieved_answer', 'N/A')
        # Truncate if too long
        if len(retrieved) > 3000:
            retrieved = retrieved[:3000] + "...\n\n[Text truncated - too long]"
        html_parts.append(f"<p style='color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.1em; white-space: pre-wrap;'>{retrieved}</p>")
        html_parts.append("</div>")
        
        # Scores
        html_parts.append("<div style='background: #3a3a3a; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ff66ff;'>")
        html_parts.append("<h4 style='color: #ffffff; margin-top: 0;'>📊 Scores:</h4>")
        html_parts.append("<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;'>")
        
        if 'alpha' in item:
            html_parts.append(f"<div><strong style='color: #ffffff;'>Alpha:</strong> <span style='color: #66b3ff;'>{item['alpha']}</span></div>")
        if 'hybrid' in item:
            html_parts.append(f"<div><strong style='color: #ffffff;'>Hybrid:</strong> <span style='color: #66b3ff;'>{item['hybrid']:.4f}</span></div>")
        if 'BM25' in item:
            html_parts.append(f"<div><strong style='color: #ffffff;'>BM25:</strong> <span style='color: #ff9966;'>{item['BM25']:.4f}</span></div>")
        if 'MARBERT' in item:
            html_parts.append(f"<div><strong style='color: #ffffff;'>MARBERT:</strong> <span style='color: #66ff66;'>{item['MARBERT']:.4f}</span></div>")
        if 'Reranker' in item:
            html_parts.append(f"<div><strong style='color: #ffffff;'>Reranker:</strong> <span style='color: #ff66ff;'>{item['Reranker']:.4f}</span></div>")
        
        html_parts.append("</div>")
        html_parts.append("</div>")
        
        # Current Evaluation Status
        if item.get('manual_rating') is not None:
            rating = item['manual_rating']
            rating_colors = {1: '#ff4444', 2: '#ff8844', 3: '#ffaa44', 4: '#88ff44', 5: '#44ff44'}
            color = rating_colors.get(rating, '#ffffff')
            html_parts.append(f"<div style='background: #1e1e1e; padding: 10px; margin: 10px 0; border-radius: 5px; border: 2px solid {color};'>")
            html_parts.append(f"<strong style='color: {color}; font-size: 1.2em;'>⭐ Current Rating: {rating}/5</strong>")
            if item.get('manual_comment'):
                html_parts.append(f"<p style='color: #d0d0d0; margin-top: 10px;'><strong>Comment:</strong> {item['manual_comment']}</p>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        return "\n".join(html_parts)
    
    def create_fast_access_buttons(self, evaluated_indices: List[int], total: int) -> str:
        """Create HTML showing completed evaluations."""
        if not evaluated_indices:
            return "<p style='color: #888;'>No evaluations completed yet.</p>"
        
        html_parts = []
        html_parts.append("<div style='background: #2b2b2b; padding: 15px; border-radius: 5px; margin: 10px 0;'>")
        html_parts.append(f"<h4 style='color: #ffffff; margin-top: 0;'>✅ Completed Evaluations ({len(evaluated_indices)}/{total}):</h4>")
        html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 5px; max-height: 300px; overflow-y: auto;'>")
        
        # Show first 100 indices to avoid overwhelming the UI
        for idx in evaluated_indices[:100]:
            item = self.data[idx]
            rating = item.get('manual_rating', 0)
            rating_colors = {1: '#ff4444', 2: '#ff8844', 3: '#ffaa44', 4: '#88ff44', 5: '#44ff44'}
            color = rating_colors.get(rating, '#ffffff')
            # Use span instead of button since we can't use onclick in Gradio HTML
            html_parts.append(
                f"<span style='background: {color}; color: white; border: none; padding: 5px 10px; "
                f"border-radius: 3px; display: inline-block; font-weight: bold; margin: 2px;'>{idx + 1}</span>"
            )
        
        if len(evaluated_indices) > 100:
            html_parts.append(f"<p style='color: #888; width: 100%; margin-top: 10px;'>... and {len(evaluated_indices) - 100} more</p>")
        
        html_parts.append("</div>")
        html_parts.append("<p style='color: #888; font-size: 0.9em; margin-top: 10px;'>Use the 'Jump to Index' field below to navigate to a specific question.</p>")
        html_parts.append("</div>")
        return "\n".join(html_parts)


def create_manual_evaluation_tab(evaluator: ManualEvaluator):
    """Create the manual evaluation tab for Gradio."""
    
    with gr.Tab("📝 Manual Evaluation"):
        gr.Markdown("""
        # 📝 Manual Evaluation of Retrieval Results
        
        Evaluate the relevance of retrieved answers to questions.
        - **5**: Retrieved answer is directly relevant and provides a solution
        - **4**: Retrieved answer is mostly relevant with minor gaps
        - **3**: Retrieved answer is somewhat relevant but incomplete
        - **2**: Retrieved answer has limited relevance
        - **1**: Retrieved chunks are not relevant
        
        Your evaluations will be saved automatically when you click Save, Next, or Previous.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Display area
                item_display = gr.HTML(label="Question & Answers")
                
                # Status
                status_text = gr.Textbox(
                    label="Status",
                    value=f"📄 Item 1 of {evaluator.get_total_count()}",
                    interactive=False
                )
                
                # Rating
                rating_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=None,
                    label="Relevance Rating (1-5)",
                    info="5 = Directly relevant with solution, 1 = Not relevant"
                )
                
                # Comment
                comment_box = gr.Textbox(
                    label="Comments (Optional)",
                    placeholder="Add any comments about this evaluation...",
                    lines=3
                )
                
                # Navigation buttons
                with gr.Row():
                    first_btn = gr.Button("⏮️ First", variant="secondary")
                    prev_btn = gr.Button("◀️ Previous", variant="secondary")
                    next_btn = gr.Button("Next ▶️", variant="secondary")
                    last_btn = gr.Button("Last ⏭️", variant="secondary")
                
                # Action buttons
                with gr.Row():
                    save_btn = gr.Button("💾 Save Evaluation", variant="primary")
                    jump_btn = gr.Button("🔢 Jump to Index", variant="secondary", visible=False)
                
                # Jump to index input
                jump_input = gr.Number(
                    label="Jump to Index (0-based)",
                    value=0,
                    minimum=0,
                    maximum=evaluator.get_total_count() - 1 if evaluator.get_total_count() > 0 else 0,
                    step=1,
                    visible=False
                )
            
            with gr.Column(scale=1):
                # Statistics
                stats_md = gr.Markdown("### 📊 Statistics\n\nLoading...")
                
                # Fast access buttons
                fast_access_html = gr.HTML(label="Fast Access")
        
        # Hidden state to track current index
        current_index_state = gr.State(value=0)
        total_count_state = gr.State(value=evaluator.get_total_count())
        
        def load_item(index: int = None):
            """Load item at given index or current index."""
            if index is not None:
                evaluator.current_index = index
            item, status, idx, total, evaluated = evaluator.navigate("")
            
            # Format display
            display_html = evaluator.format_item_display(item)
            
            # Get current rating and comment
            current_rating = item.get('manual_rating') if item else None
            current_comment = item.get('manual_comment', '') if item else ''
            
            # Update stats
            evaluated_count = evaluator.get_evaluated_count()
            percentage = (evaluated_count/total*100) if total > 0 else 0
            stats_text = f"""
            ### 📊 Statistics
            
            - **Total Items**: {total}
            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
            - **Remaining**: {total - evaluated_count}
            - **Current Position**: {idx + 1} / {total}
            """
            
            # Fast access buttons
            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
            
            return (
                display_html,
                status,
                current_rating,
                current_comment,
                stats_text,
                fast_access,
                idx,
                total
            )
        
        def save_and_next(rating, comment):
            """Save evaluation and move to next."""
            if rating is not None:
                evaluator.update_evaluation(int(rating), comment)
                evaluator.save_data()
            
            # Move to next
            item, status, idx, total, evaluated = evaluator.navigate("next")
            display_html = evaluator.format_item_display(item)
            current_rating = item.get('manual_rating') if item else None
            current_comment = item.get('manual_comment', '') if item else ''
            
            evaluated_count = evaluator.get_evaluated_count()
            percentage = (evaluated_count/total*100) if total > 0 else 0
            stats_text = f"""
            ### 📊 Statistics
            
            - **Total Items**: {total}
            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
            - **Remaining**: {total - evaluated_count}
            - **Current Position**: {idx + 1} / {total}
            """
            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
            
            return (
                display_html,
                status,
                current_rating,
                current_comment,
                stats_text,
                fast_access,
                idx,
                total,
                "✅ Saved and moved to next!"
            )
        
        def save_evaluation(rating, comment):
            """Save evaluation without moving."""
            if rating is not None:
                evaluator.update_evaluation(int(rating), comment)
                return evaluator.save_data()
            return "⚠️ Please select a rating first"
        
        def navigate(direction):
            """Navigate to next/previous/first/last."""
            item, status, idx, total, evaluated = evaluator.navigate(direction)
            display_html = evaluator.format_item_display(item)
            current_rating = item.get('manual_rating') if item else None
            current_comment = item.get('manual_comment', '') if item else ''
            
            evaluated_count = evaluator.get_evaluated_count()
            percentage = (evaluated_count/total*100) if total > 0 else 0
            stats_text = f"""
            ### 📊 Statistics
            
            - **Total Items**: {total}
            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
            - **Remaining**: {total - evaluated_count}
            - **Current Position**: {idx + 1} / {total}
            """
            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
            
            return (
                display_html,
                status,
                current_rating,
                current_comment,
                stats_text,
                fast_access,
                idx,
                total
            )
        
        def jump_to(jump_idx):
            """Jump to specific index."""
            item, status, idx, total, evaluated = evaluator.jump_to_index(int(jump_idx))
            display_html = evaluator.format_item_display(item)
            current_rating = item.get('manual_rating') if item else None
            current_comment = item.get('manual_comment', '') if item else ''
            
            evaluated_count = evaluator.get_evaluated_count()
            percentage = (evaluated_count/total*100) if total > 0 else 0
            stats_text = f"""
            ### 📊 Statistics
            
            - **Total Items**: {total}
            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
            - **Remaining**: {total - evaluated_count}
            - **Current Position**: {idx + 1} / {total}
            """
            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
            
            return (
                display_html,
                status,
                current_rating,
                current_comment,
                stats_text,
                fast_access,
                idx,
                total
            )
        
        # Initial load
        initial_outputs = load_item(0)
        
        # Event handlers
        save_btn.click(
            fn=save_evaluation,
            inputs=[rating_slider, comment_box],
            outputs=[status_text]
        )
        
        next_btn.click(
            fn=lambda r, c: save_and_next(r, c),
            inputs=[rating_slider, comment_box],
            outputs=[
                item_display,
                status_text,
                rating_slider,
                comment_box,
                stats_md,
                fast_access_html,
                current_index_state,
                total_count_state,
                status_text
            ]
        )
        
        prev_btn.click(
            fn=lambda: navigate("previous"),
            outputs=[
                item_display,
                status_text,
                rating_slider,
                comment_box,
                stats_md,
                fast_access_html,
                current_index_state,
                total_count_state
            ]
        )
        
        first_btn.click(
            fn=lambda: navigate("first"),
            outputs=[
                item_display,
                status_text,
                rating_slider,
                comment_box,
                stats_md,
                fast_access_html,
                current_index_state,
                total_count_state
            ]
        )
        
        last_btn.click(
            fn=lambda: navigate("last"),
            outputs=[
                item_display,
                status_text,
                rating_slider,
                comment_box,
                stats_md,
                fast_access_html,
                current_index_state,
                total_count_state
            ]
        )
        
        # Update display when rating or comment changes (auto-save on next/prev)
        rating_slider.change(
            fn=lambda r, c: (evaluator.update_evaluation(int(r) if r else None, c), "")[1] if r else "",
            inputs=[rating_slider, comment_box],
            outputs=[status_text]
        )
        
        # Load initial data on tab load
        def on_tab_load():
            return load_item(0)
        
        # Use load event to initialize
        item_display.load(
            fn=on_tab_load,
            inputs=[],
            outputs=[
                item_display,
                status_text,
                rating_slider,
                comment_box,
                stats_md,
                fast_access_html,
                current_index_state,
                total_count_state
            ]
        )

