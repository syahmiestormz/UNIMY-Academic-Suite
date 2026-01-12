import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re

# Page Configuration
st.set_page_config(page_title="UNIMY Academic Suite", layout="wide", page_icon="🎓")

# --- CSS for Styling ---
st.markdown("""
<style>
    .main-header {font-size: 2rem; color: #4A90E2; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #333;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;}
    .success-box {background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;}
    .warning-box {background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px;}
    .report-area {font-family: 'Times New Roman'; background: white; padding: 20px; border: 1px solid #ccc;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS (SHARED)
# ==========================================

def find_header_row(df, keywords):
    """Finds row index with specific keywords."""
    for idx, row in df.iterrows():
        row_str = row.astype(str).str.cat(sep=' ').lower()
        if all(k.lower() in row_str for k in keywords):
            return idx
    return -1

def clean_percentage(val):
    """Normalizes percentage values to 0-100 float."""
    try:
        if isinstance(val, str): val = val.replace('%', '')
        num = float(val)
        return num * 100 if num <= 1.0 else num
    except: return 0.0

def get_smart_recommendation(clo_name, failure_rate):
    """Generates CQI actions based on context."""
    if failure_rate < 15: return "Maintain current teaching methods."
    context = str(clo_name).lower()
    if "drawing" in context: return "Conduct extra studio sessions with live demonstrations."
    if "calculation" in context: return "Provide remedial drills focusing on step-by-step methods."
    if "software" in context: return "Organize extra lab tutorials for software proficiency."
    return "Review assessment difficulty and conduct revision classes."

# ==========================================
# LECTURER MODULE: SUBJECT ANALYTICS
# ==========================================

def parse_campusone_file(uploaded_file):
    """Parses raw CampusOne Excel export."""
    try:
        df = pd.read_excel(uploaded_file, header=None)
        info = {"code": "Unknown", "name": "Unknown", "semester": "Unknown", "lecturer": "Unknown"}
        
        # Extract Metadata
        for r in range(min(20, len(df))):
            row_str = df.iloc[r].astype(str).str.cat(sep=' ')
            if "Subject" in row_str and ":" in row_str:
                parts = row_str.split(":", 1)[1].strip()
                if "-" in parts:
                    info["code"] = parts.split("-")[0].strip()
                    info["name"] = parts.split("-", 1)[1].strip()
                else: info["name"] = parts
            if "Semester" in row_str and ":" in row_str:
                info["semester"] = row_str.split(":", 1)[1].strip()
            if "Lecturer" in row_str and ":" in row_str:
                info["lecturer"] = row_str.split(":", 1)[1].strip()

        # Find Header
        header_idx = find_header_row(df, ["Student No.", "Student Name"])
        if header_idx == -1: return None, None, "Header not found."

        # Extract Data
        df.columns = df.iloc[header_idx]
        data_df = df.iloc[header_idx+1:].reset_index(drop=True)
        # Filter valid students
        data_df = data_df[pd.to_numeric(data_df.iloc[:, 0], errors='coerce').notna()]
        
        # Identify Columns
        potential_cols = [str(c).strip() for c in data_df.columns if str(c).strip() not in 
                          ["No", "Student No.", "Student Name", "Enrollment No.", "Programme", "Intake", "Assessment Mark", "Total Mark", "Grade", "Point", "Result", "Hold", "Note", "nan"]]
        
        return data_df, info, potential_cols
    except Exception as e: return None, None, str(e)

def calculate_subject_performance(data_df, config_map, plo_map):
    """Calculates CLO/PLO attainment from raw marks."""
    results = []
    total_gpa, count_gpa = 0, 0
    grades_map = {'A+': 4.0, 'A': 4.0, 'A-': 3.67, 'B+': 3.33, 'B': 3.0, 'B-': 2.67, 'C+': 2.33, 'C': 2.0, 'D': 1.0, 'F': 0.0}

    for _, row in data_df.iterrows():
        s_id = str(row.get("Student No.", "")).strip()
        s_name = str(row.get("Student Name", "")).strip()
        grade = str(row.get("Grade", "")).strip().upper()
        
        if grade in grades_map:
            total_gpa += grades_map[grade]
            count_gpa += 1
            
        student_clos = {}
        for col, cfg in config_map.items():
            try:
                raw = pd.to_numeric(row.get(col, 0), errors='coerce')
                if pd.isna(raw): raw = 0
                # Calc weighted score
                contrib = (raw / cfg['full']) * cfg['weight']
                clo = cfg['clo']
                
                if clo not in student_clos: student_clos[clo] = {'earned': 0, 'weight': 0}
                student_clos[clo]['earned'] += contrib
                student_clos[clo]['weight'] += cfg['weight']
            except: pass
            
        # Finalize Student
        res = {'Student ID': s_id, 'Student Name': s_name, 'Grade': grade, 'Total': 0}
        total_score = 0
        
        for clo, vals in student_clos.items():
            if vals['weight'] > 0:
                perc = (vals['earned'] / vals['weight']) * 100
                res[clo] = perc
                total_score += vals['earned'] # Sum of weighted parts
        
        res['Total'] = total_score
        results.append(res)

    df_res = pd.DataFrame(results)
    avg_gpa = total_gpa / count_gpa if count_gpa > 0 else 0
    
    # CLO Stats
    clo_stats = []
    clo_cols = sorted([c for c in df_res.columns if "CLO" in c])
    
    for clo in clo_cols:
        avg = df_res[clo].mean()
        pass_count = len(df_res[df_res[clo] >= 50])
        pass_rate = (pass_count / len(df_res)) * 100
        
        # Mapping to PLO
        mapped_plo = plo_map.get(clo, "-")
        
        clo_stats.append({
            "CLO": clo,
            "PLO": mapped_plo,
            "Overall %": avg,
            "Student Pass Rate (%)": pass_rate,
            "Status": "YES" if avg >= 50 else "NO"
        })
        
    return df_res, pd.DataFrame(clo_stats), avg_gpa

def generate_audit_excel(info, df_marks, df_clo, plo_map):
    """Generates the standardized Excel for the Dean."""
    # Use openpyxl for compatibility
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Setup Sheet
        pd.DataFrame([info]).to_excel(writer, sheet_name='Setup', index=False)
        
        # 2. Marks Sheet
        df_marks.to_excel(writer, sheet_name='Table 1 - Marks', index=False)
        
        # 3. CLO Sheet
        df_clo.to_excel(writer, sheet_name='Table 2 - CLO Analysis', index=False)
        
        # 4. PLO Sheet (Derived)
        plo_rows = []
        for _, row in df_clo.iterrows():
            if row['PLO'] != '-':
                plo_rows.append({"CLO": row['CLO'], "Mapped PLO": row['PLO'], "Score (%)": row['Overall %']})
        pd.DataFrame(plo_rows).to_excel(writer, sheet_name='Table 3 - PLO Analysis', index=False)
        
    return output.getvalue()

# ==========================================
# DEAN MODULE: PROGRAMME ANALYTICS
# ==========================================

def process_dean_files(uploaded_files):
    """Aggregates multiple 'Processed Master Excel' files."""
    all_student_records = []
    course_summaries = []
    
    for f in uploaded_files:
        try:
            xls = pd.read_excel(f, sheet_name=None)
            
            # Read Info
            if 'Setup' in xls:
                info_df = xls['Setup']
                course_code = info_df['code'].iloc[0] if 'code' in info_df.columns else "Unknown"
                course_name = info_df['name'].iloc[0] if 'name' in info_df.columns else "Unknown"
            else: continue
            
            # Read Marks & PLO Data
            if 'Table 1 - Marks' in xls and 'Table 2 - CLO Analysis' in xls:
                df_marks = xls['Table 1 - Marks']
                df_clo = xls['Table 2 - CLO Analysis']
                
                # Create PLO mapping from Table 2
                clo_to_plo = dict(zip(df_clo['CLO'], df_clo['PLO']))
                
                # Calculate Pass Rate
                pass_rate = (len(df_marks[df_marks['Total'] >= 50]) / len(df_marks)) * 100
                course_summaries.append({'code': course_code, 'name': course_name, 'pass_rate': pass_rate})
                
                # Process Each Student
                for _, row in df_marks.iterrows():
                    s_id = str(row['Student ID'])
                    if pd.isna(s_id) or s_id == 'nan': continue
                    
                    # Convert CLO scores to PLO scores for this student
                    student_plos = {}
                    for clo, plo in clo_to_plo.items():
                        if plo != '-' and clo in row:
                            score = row[clo]
                            if plo not in student_plos: student_plos[plo] = []
                            student_plos[plo].append(score)
                    
                    # Avg if multiple CLOs map to same PLO
                    final_plos = {k: sum(v)/len(v) for k, v in student_plos.items()}
                    
                    all_student_records.append({
                        'Student ID': s_id,
                        'Student Name': row['Student Name'],
                        'Course Code': course_code,
                        'Course Name': course_name,
                        'PLO_Data': final_plos
                    })
                    
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")
            
    return pd.DataFrame(all_student_records), course_summaries

# ==========================================
# MAIN APP UI
# ==========================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/university.png", width=50)
    st.markdown("## UNIMY Academic Suite")
    st.markdown("---")
    role = st.radio("Select Role:", ["👨‍🏫 Lecturer (Subject)", "🎓 Dean (Programme)"])
    st.markdown("---")
    st.caption("© 2026 UNIMY Quality Assurance")

if role == "👨‍🏫 Lecturer (Subject)":
    st.title("👨‍🏫 Subject Assessment (Lecturer Portal)")
    st.markdown("Upload raw CampusOne data to generate your Subject Audit Report.")
    
    uploaded_file = st.file_uploader("Upload CampusOne Raw Excel", type=['xlsx'])
    
    if uploaded_file:
        data_df, info, potential_cols = parse_campusone_file(uploaded_file)
        
        if data_df is not None:
            st.success(f"Loaded: {info['code']} - {info['name']} ({len(data_df)} Students)")
            
            with st.form("lecturer_config"):
                st.subheader("1. Configure Assessments")
                configs = {}
                cols = st.columns(3)
                for i, col in enumerate(potential_cols):
                    with cols[i%3]:
                        if st.checkbox(f"Use '{col}'", value=True, key=f"use_{i}"):
                            clo = st.selectbox("CLO", ["CLO 1", "CLO 2", "CLO 3", "CLO 4"], key=f"c_{i}")
                            cat = st.selectbox("Type", ["CA", "FP", "FE"], key=f"t_{i}")
                            w = st.number_input("Weight %", 20.0, key=f"w_{i}")
                            f = st.number_input("Full Mark", 100.0, key=f"f_{i}")
                            configs[col] = {'clo': clo, 'cat': cat, 'weight': w, 'full': f}
                
                st.subheader("2. Map CLOs to PLOs")
                c1, c2, c3, c4 = st.columns(4)
                plo_map = {}
                plo_opts = ["-"] + [f"PLO {i}" for i in range(1, 13)]
                with c1: plo_map["CLO 1"] = st.selectbox("CLO 1 ->", plo_opts, index=1)
                with c2: plo_map["CLO 2"] = st.selectbox("CLO 2 ->", plo_opts, index=2)
                with c3: plo_map["CLO 3"] = st.selectbox("CLO 3 ->", plo_opts, index=3)
                with c4: plo_map["CLO 4"] = st.selectbox("CLO 4 ->", plo_opts, index=0)
                
                run_calc = st.form_submit_button("🚀 Run Analysis")
            
            if run_calc and configs:
                df_marks, df_clo, gpa = calculate_subject_performance(data_df, configs, plo_map)
                
                # TABS
                t1, t2, t3, t4 = st.tabs(["Student Results", "CLO Analysis", "ESPAR Report", "⬇️ Audit File"])
                
                with t1:
                    st.dataframe(df_marks.style.format("{:.1f}", subset=[c for c in df_marks.columns if "CLO" in c or "Total" in c]), use_container_width=True)
                
                with t2:
                    st.dataframe(df_clo.style.format("{:.1f}", subset=["Overall %", "Student Pass Rate (%)"]), use_container_width=True)
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.bar(df_clo['CLO'], df_clo['Overall %'], color=['#4CAF50' if x>=50 else '#F44336' for x in df_clo['Overall %']])
                    ax.axhline(50, color='black', linestyle='--')
                    st.pyplot(fig)
                    
                with t3:
                    pass_rate = (len(df_marks[df_marks['Total']>=50])/len(df_marks))*100
                    report = f"""**SUBJECT PERFORMANCE REPORT**
Code: {info['code']}
Pass Rate: {pass_rate:.1f}%
Average GPA: {gpa:.2f}

**CLO ANALYSIS**
"""
                    for _, row in df_clo.iterrows():
                        report += f"- {row['CLO']}: {row['Overall %']:.1f}% ({'MET' if row['Overall %']>=50 else 'NOT MET'})\n"
                    
                    report += "\n**CQI ACTIONS**\n"
                    weak = df_clo[df_clo['Overall %'] < 50]
                    if not weak.empty:
                        for _, w in weak.iterrows():
                            rec = get_smart_recommendation(info['name'], 100-w['Student Pass Rate (%)'])
                            report += f"| {w['CLO']} | Low Attainment | {rec} |\n"
                    else:
                        report += "No critical failures observed."
                    
                    st.text_area("Copy for ESPAR", report, height=300)
                    
                with t4:
                    st.success("Download this file and submit it to your Dean.")
                    excel_data = generate_audit_excel(info, df_marks, df_clo, plo_map)
                    st.download_button("Download Processed Master Excel", excel_data, f"Processed_{info['code']}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif role == "🎓 Dean (Programme)":
    st.title("🎓 Programme Analytics (Dean Portal)")
    st.markdown("Upload multiple **Processed Master Excel** files (from Lecturers) to generate the Programme Report.")
    
    uploaded_files = st.file_uploader("Upload Subject Files", accept_multiple_files=True, type=['xlsx'])
    
    if uploaded_files:
        df_master, courses = process_dean_files(uploaded_files)
        
        if not df_master.empty:
            t1, t2, t3 = st.tabs(["Programme Heatmap", "Student Scorecard", "📄 Generate ESPAR"])
            
            # Aggregation
            all_plo_vals = {}
            for _, row in df_master.iterrows():
                for plo, val in row['PLO_Data'].items():
                    if plo not in all_plo_vals: all_plo_vals[plo] = []
                    all_plo_vals[plo].append(val)
            
            plo_avgs = {k: sum(v)/len(v) for k, v in all_plo_vals.items()}
            sorted_plos = sorted(plo_avgs.keys(), key=lambda x: int(x.split(' ')[-1]))
            
            with t1:
                st.subheader("Programme PLO Attainment")
                c1, c2 = st.columns(2)
                c1.metric("Total Students", len(df_master['Student ID'].unique()))
                c2.metric("Avg PLO Achievement", f"{np.mean(list(plo_avgs.values())):.1f}%")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(sorted_plos, [plo_avgs[k] for k in sorted_plos], color='#4A90E2')
                ax.axhline(50, color='red', linestyle='--')
                st.pyplot(fig)
                
            with t2:
                st.subheader("Individual Student Tracker")
                s_list = df_master['Student Name'].unique()
                sel_s = st.selectbox("Select Student", sorted(s_list))
                
                s_data = df_master[df_master['Student Name'] == sel_s]
                st.write(f"**Course History for {sel_s}:**")
                
                flat_data = []
                for _, r in s_data.iterrows():
                    d = {'Course': r['Course Code']}
                    d.update(r['PLO_Data'])
                    flat_data.append(d)
                
                st.dataframe(pd.DataFrame(flat_data).style.format("{:.1f}"), use_container_width=True)
                
            with t3:
                st.subheader("📄 Programme ESPAR Generator")
                avg_pass = sum(c['pass_rate'] for c in courses)/len(courses) if courses else 0
                weak_courses = [c['code'] for c in courses if c['pass_rate'] < 85]
                
                report = f"""**1.0 EXECUTIVE SUMMARY**
Overall Status: Satisfactory. The cohort achieved an average pass rate of {avg_pass:.1f}%.
Key Strength: Students excelled in {max(plo_avgs, key=plo_avgs.get)} ({plo_avgs[max(plo_avgs, key=plo_avgs.get)]:.1f}%).
Key Weakness: {f"Attention needed in {min(plo_avgs, key=plo_avgs.get)}." if plo_avgs else "None."}

**3.0 ANALYSIS OF LEARNING OUTCOMES**
**3.1 Course Learning Outcomes (CLO)**
Overall Pass Rate: {avg_pass:.1f}%
High Failure Courses: {", ".join(weak_courses) if weak_courses else "None"}

**3.2 Programme Learning Outcomes (PLO)**
| PLO | Score | Status |
|---|---|---|
"""
                for plo in sorted_plos:
                    report += f"| {plo} | {plo_avgs[plo]:.1f}% | {'ACHIEVED' if plo_avgs[plo]>=50 else 'ATTENTION'} |\n"
                
                st.text_area("Programme Report Text", report, height=500)