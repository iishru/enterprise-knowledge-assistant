# make_sample_docs.py - generates the 8 fictional "Nimbus Technologies" policy documents.
# Run: python data/make_sample_docs.py   (needs reportlab and python-docx from requirements-dev.txt)
# Each document is a list of sections; a section is (heading, [items]).
# An item is either a paragraph (str) or a bullet list (list of str).
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "sample_docs"

DOCS = [
    {
        "file": "leave_policy.pdf",
        "title": "Nimbus Technologies - Leave Policy",
        "sections": [
            ("1. Purpose and Scope", [
                "This policy explains the leave entitlements of all permanent employees of Nimbus Technologies Pvt. Ltd. "
                "The leave year runs from 1 January to 31 December. All leave is applied for and approved through the NimbusHR portal.",
            ]),
            ("2. Types of Leave", [
                ["Casual Leave: 12 days per year. Casual leave cannot be combined with more than 3 consecutive days of other leave, and unused casual leave lapses on 31 December.",
                 "Sick Leave: 10 days per year. A medical certificate is required if sick leave is taken for more than 2 consecutive days. Sick leave cannot be carried forward.",
                 "Earned Leave: 18 days per year, credited at 1.5 days per month. Earned leave can be carried forward up to a maximum balance of 30 days. Anything above 30 days is encashed in the January payroll.",
                 "Maternity Leave: 26 weeks of paid leave for the first two children.",
                 "Paternity Leave: 10 working days, to be taken within 6 months of the child's birth.",
                 "Bereavement Leave: 5 days on the death of an immediate family member (spouse, child, parent, sibling).",
                 "Public Holidays: 12 company holidays per year. The list is published on NimbusHR every December."],
            ]),
            ("3. Employees on Probation", [
                "Employees on probation (the first 6 months) receive casual leave and sick leave on a pro-rata basis. "
                "They do not earn Earned Leave until they are confirmed. Maternity and paternity leave apply only after confirmation.",
            ]),
            ("4. How to Apply", [
                ["Apply on NimbusHR at least 3 working days in advance for any leave longer than 2 days.",
                 "For leave of 2 days or less, apply as early as possible; sick leave may be applied for on the same day.",
                 "Your reporting manager must approve or reject the request within 2 working days. If there is no response, the request is escalated to the department head.",
                 "Leave taken without approval is treated as Loss of Pay."],
            ]),
            ("5. Compensatory Off", [
                "An employee who works on a public holiday or weekend with written manager approval earns one compensatory off for each day worked. "
                "Compensatory offs must be used within 60 days of being earned, otherwise they lapse.",
            ]),
            ("6. Leave at Exit", [
                "Unused earned leave is encashed at the time of exit, calculated on basic salary divided by 26 days. "
                "Unused casual leave and sick leave are not encashed.",
            ]),
            ("7. Questions", [
                "Contact the HR helpdesk at hr@nimbus.example or extension 2100.",
            ]),
        ],
    },
    {
        "file": "it_security_policy.pdf",
        "title": "Nimbus Technologies - IT Security Policy",
        "sections": [
            ("1. Purpose", [
                "This policy protects Nimbus Technologies systems and data. It applies to every employee, contractor and intern who uses company devices or accounts.",
            ]),
            ("2. Passwords and Multi-Factor Authentication", [
                ["Passwords must be at least 14 characters long and include upper case, lower case, a number and a symbol.",
                 "Passwords must be changed every 90 days. You cannot reuse any of your last 5 passwords.",
                 "Multi-factor authentication (MFA) is mandatory for email, VPN and all cloud tools. Use the Nimbus Authenticator app on your phone.",
                 "Never share your password or MFA code with anyone, including the IT helpdesk."],
            ]),
            ("3. Devices", [
                ["All company laptops are encrypted with BitLocker. Do not disable encryption.",
                 "Screens must lock automatically after 5 minutes of inactivity. Lock your screen manually whenever you leave your desk.",
                 "USB storage devices are blocked by default. If you need an exception, raise a request on NimbusDesk with your manager's approval.",
                 "Install software only from the Software Center. Installing unapproved software is a policy violation."],
            ]),
            ("4. Remote Access and Wi-Fi", [
                "The NimbusSecure VPN must be switched on whenever you work from home, a cafe, an airport or any public Wi-Fi network. "
                "Company data must never be accessed on public Wi-Fi without the VPN.",
            ]),
            ("5. Data Classification", [
                ["Public: information approved for anyone, for example press releases.",
                 "Internal: for employees only, for example this policy.",
                 "Confidential: client contracts, financial data and source code. Share only with people who need it for their work.",
                 "Restricted: employee personal data, credentials and security audit reports. Must be encrypted and can only be shared with named approved people."],
            ]),
            ("6. Reporting Incidents", [
                "Report a lost or stolen device, a suspected phishing email or any suspected data breach within 1 hour to security@nimbus.example "
                "or call the security desk on extension 5555. Use the Report Phishing button in your email client for suspicious emails. "
                "You will not be penalised for reporting in good faith.",
            ]),
            ("7. Training", [
                "Every employee must complete the annual security awareness training by 31 March. New joiners must complete it within their first 2 weeks.",
            ]),
        ],
    },
    {
        "file": "travel_expense_policy.pdf",
        "title": "Nimbus Technologies - Travel and Expense Reimbursement Policy",
        "sections": [
            ("1. Scope", [
                "This policy covers business travel within India and abroad and the reimbursement of related expenses.",
            ]),
            ("2. City Tiers and Limits", [
                "Cities are grouped into three tiers. The limits below apply per person.",
                ["Tier 1 (Mumbai, Delhi, Bengaluru): hotel up to Rs 6,500 per night; meal allowance Rs 1,200 per day.",
                 "Tier 2 (Pune, Hyderabad, Chennai, Kolkata, Ahmedabad): hotel up to Rs 4,500 per night; meal allowance Rs 900 per day.",
                 "Tier 3 (all other cities): hotel up to Rs 3,000 per night; meal allowance Rs 700 per day."],
                "Expenses above the limit are not reimbursed unless the department head approves in writing before the trip.",
            ]),
            ("3. Transport", [
                ["Flights: economy class only. Book at least 7 days in advance through the company travel desk.",
                 "Trains: 2nd AC or equivalent.",
                 "Local cabs: reimbursed with a receipt.",
                 "Personal vehicle: Rs 9 per km for a car and Rs 4 per km for a two-wheeler, with a trip log."],
            ]),
            ("4. Approvals", [
                ["Trips costing up to Rs 25,000 need your reporting manager's approval.",
                 "Trips costing more than Rs 25,000 also need the department head's approval.",
                 "International travel needs approval from a Vice President before booking."],
            ]),
            ("5. Travel Advance", [
                "You may request an advance of up to 70% of the estimated trip cost at least 5 working days before travel. "
                "The advance must be settled within 15 days of returning.",
            ]),
            ("6. Submitting a Claim", [
                ["Submit your claim on the expense portal within 15 days of returning from the trip. Late claims are rejected.",
                 "A receipt is required for every expense above Rs 500.",
                 "Approved claims are paid with payroll within 10 working days.",
                 "Alcohol, personal entertainment, fines and hotel minibar charges are never reimbursed."],
            ]),
        ],
    },
    {
        "file": "employee_onboarding_guide.docx",
        "title": "Nimbus Technologies - Employee Onboarding Guide",
        "sections": [
            ("Welcome", [
                "Welcome to Nimbus Technologies. This guide helps you through your first weeks. Your first 6 months are your probation period.",
            ]),
            ("Day 1", [
                ["Report to the Floor 3 reception at 10:00 AM for the HR orientation.",
                 "Collect your ID card from the security desk on the ground floor. Bring one photo ID.",
                 "Collect your laptop from the IT desk on Floor 2. IT will help you set up your password and MFA.",
                 "You will be introduced to your onboarding buddy, who is a teammate that answers your day-to-day questions for the first 3 months."],
            ]),
            ("Week 1 Checklist", [
                ["Sign in to NimbusHR and complete your profile and bank details.",
                 "Join NimbusChat, the company chat tool, and your team's channels.",
                 "Get access to Jira, Confluence and GitHub Enterprise by raising a request on NimbusDesk.",
                 "Meet your manager to agree your 30-60-90 day plan.",
                 "Start the mandatory training modules: IT Security, Code of Conduct and POSH awareness."],
            ]),
            ("Working Hours", [
                "Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday. Core hours are 11:00 AM to 4:00 PM, when everyone is expected to be available.",
            ]),
            ("Mandatory Training", [
                "All mandatory training modules must be completed within the first 2 weeks of joining.",
            ]),
            ("Salary and Payroll", [
                "Salary is paid on the last working day of every month. Payslips are on NimbusHR. Your first salary may be pro-rated based on your joining date.",
            ]),
            ("Useful Contacts", [
                ["HR helpdesk: hr@nimbus.example, extension 2100.",
                 "IT helpdesk: extension 5000, or raise a ticket on NimbusDesk.",
                 "Security desk: extension 5555.",
                 "Facilities and cafeteria: extension 3300."],
            ]),
            ("Probation Review", [
                "Your manager reviews your progress at 3 months and at 6 months. Confirmation is based on the 6-month review.",
            ]),
        ],
    },
    {
        "file": "work_from_home_policy.docx",
        "title": "Nimbus Technologies - Work From Home Policy",
        "sections": [
            ("1. Hybrid Model", [
                "Nimbus follows a hybrid model. Confirmed employees work from the office on 3 days a week (Tuesday, Wednesday and Thursday) "
                "and may work from home on Monday and Friday.",
            ]),
            ("2. Eligibility", [
                ["Employees on probation work from the office all 5 days.",
                 "Employees in client-facing support roles follow the schedule set by their team lead.",
                 "Your manager can pause your work-from-home days if performance or availability is a concern."],
            ]),
            ("3. Extra Work-From-Home Days", [
                "To work from home on a Tuesday, Wednesday or Thursday, request it on NimbusHR at least 1 day in advance and get your manager's approval. "
                "Each employee can take up to 6 extra work-from-home days per quarter.",
            ]),
            ("4. Working From Another City or Country", [
                "Working remotely from a city other than your base location needs approval from your manager and HR. "
                "The maximum is 15 days per calendar year. Working from another country is not allowed because of tax and legal rules.",
            ]),
            ("5. Availability", [
                "On work-from-home days you must be online on NimbusChat during core hours (11:00 AM to 4:00 PM) and attend all scheduled meetings with your camera on when possible.",
            ]),
            ("6. Allowances", [
                ["Internet allowance: Rs 2,000 per month for confirmed employees on the hybrid schedule.",
                 "One-time home office setup allowance: Rs 10,000 for a desk, chair or monitor. Submit bills on the expense portal within 30 days of purchase."],
            ]),
            ("7. Security", [
                "The NimbusSecure VPN must be on while you work from home. Follow the IT Security Policy at all times. Do not let family members use your company laptop.",
            ]),
        ],
    },
    {
        "file": "code_of_conduct.md",
        "title": "Nimbus Technologies - Code of Conduct",
        "sections": [
            ("Our Values", [
                "We act with integrity, respect colleagues and clients, and protect the company's reputation. This code applies to all employees, contractors and interns.",
            ]),
            ("Respect and Anti-Harassment", [
                ["Nimbus has zero tolerance for harassment, bullying and discrimination based on gender, religion, caste, age, disability or any other personal characteristic.",
                 "Sexual harassment complaints are handled by the Internal Committee (IC) under the POSH Act. Contact the IC at ic@nimbus.example.",
                 "Retaliation against anyone who raises a concern in good faith is a serious violation and can lead to dismissal."],
            ]),
            ("Gifts and Hospitality", [
                "Employees must not accept gifts worth more than Rs 2,000 from clients or vendors. Cash or cash equivalents, such as gift cards, must always be refused. "
                "Report any gift above the limit to your manager within 5 working days.",
            ]),
            ("Conflict of Interest", [
                ["Disclose any personal or financial interest in a vendor, client or competitor to your manager and HR.",
                 "Outside employment, freelancing or running a business needs written approval from HR before you start."],
            ]),
            ("Confidentiality", [
                "Do not share client data, source code or internal documents outside the company. This duty continues after you leave Nimbus. "
                "Do not post confidential information or comment on behalf of the company on social media.",
            ]),
            ("Raising a Concern", [
                ["Talk to your manager or HR first.",
                 "You can also write to the Ethics Helpline at ethics@nimbus.example or call extension 7000. Reports can be made anonymously.",
                 "Every report is acknowledged within 3 working days, and investigations are normally completed within 30 days."],
            ]),
            ("Consequences", [
                "Breaking this code can lead to a warning, suspension, termination or legal action, depending on how serious it is.",
            ]),
        ],
    },
    {
        "file": "it_helpdesk_faq.md",
        "title": "Nimbus Technologies - IT Helpdesk FAQ",
        "sections": [
            ("How do I contact the IT helpdesk?", [
                "Raise a ticket on the NimbusDesk portal, email it-help@nimbus.example, or call extension 5000. "
                "The helpdesk is open from 8:00 AM to 8:00 PM, Monday to Saturday. Critical (P1) issues are handled 24x7.",
            ]),
            ("What are the ticket response and resolution times?", [
                ["P1 (Critical, for example a complete outage or a security breach): response within 1 hour, resolution within 4 hours.",
                 "P2 (High, for example one team cannot work): response within 4 hours, resolution within 1 business day.",
                 "P3 (Medium, for example a single user issue): response within 1 business day, resolution within 3 business days.",
                 "P4 (Low, for example a request or enhancement): response within 2 business days, resolution within 5 business days."],
            ]),
            ("I forgot my password. What should I do?", [
                "Use the self-service password portal and verify with the Nimbus Authenticator app. If your account is locked after 5 wrong attempts, "
                "it unlocks automatically after 30 minutes, or you can call extension 5000 to unlock it sooner.",
            ]),
            ("My laptop is slow or broken.", [
                ["Restart the laptop first. Many issues are fixed by an update and a restart.",
                 "If the problem continues, raise a P3 ticket on NimbusDesk.",
                 "Laptops are refreshed every 3 years (36 months). Your laptop is also replaced earlier if IT confirms it cannot be repaired."],
            ]),
            ("How do I get access to a tool or shared folder?", [
                "Raise an access request on NimbusDesk. Your manager must approve it first, and IT fulfils approved requests within 2 business days.",
            ]),
            ("How do I install new software?", [
                "Open the Software Center on your laptop and choose from the approved list. If the software you need is not listed, raise a software request on NimbusDesk. "
                "The security team reviews new software requests, which takes up to 5 business days.",
            ]),
            ("The VPN is not connecting.", [
                ["Check that your internet works and that the date and time on your laptop are correct.",
                 "Sign out of NimbusSecure, sign in again and approve the MFA prompt.",
                 "If it still fails, raise a P2 ticket and include a screenshot of the error."],
            ]),
            ("I lost my laptop or phone.", [
                "Report it within 1 hour to security@nimbus.example or on extension 5555. This is required by the IT Security Policy.",
            ]),
        ],
    },
    {
        "file": "performance_review_process.md",
        "title": "Nimbus Technologies - Performance Review Process",
        "sections": [
            ("Review Cycle", [
                "The performance year runs from 1 April to 31 March. There are two formal reviews: a mid-year check-in in October and an annual review in April.",
            ]),
            ("Setting Goals", [
                "Goals (called OKRs) are set in April, or within 30 days of joining for new employees. Each employee has 3 to 5 goals, agreed with the manager and recorded on NimbusHR.",
            ]),
            ("Steps in the Annual Review", [
                ["Self-assessment: the employee completes it on NimbusHR within 5 working days of the cycle opening.",
                 "Peer feedback: at least 3 peers give feedback (360 feedback). The employee suggests peers and the manager approves the list.",
                 "Manager assessment: the manager writes the assessment and proposes a rating.",
                 "Calibration: department heads review ratings together to keep them fair across teams.",
                 "Feedback conversation: the manager shares the final rating and next steps."],
            ]),
            ("Rating Scale", [
                ["5 - Outstanding: far exceeds expectations.",
                 "4 - Exceeds Expectations.",
                 "3 - Meets Expectations: the expected performance level.",
                 "2 - Needs Improvement.",
                 "1 - Unsatisfactory."],
            ]),
            ("Rewards and Promotion", [
                "Salary increments take effect on 1 April. Only employees with a rating of 4 or 5 are considered for promotion, "
                "and they need at least 12 months in their current role. Employees who joined less than 6 months before the cycle ends are not rated, but receive written feedback.",
            ]),
            ("Performance Improvement Plan", [
                "An employee rated 1, or rated 2 in two consecutive cycles, is placed on a Performance Improvement Plan (PIP) lasting 60 days, with weekly check-ins with the manager and HR.",
            ]),
            ("Appeals", [
                "If you disagree with your rating, you can submit an appeal to HR within 10 working days of the feedback conversation. A review panel decides within 15 working days.",
            ]),
        ],
    },
]


def write_markdown(doc: dict, path: Path) -> None:
    lines = [f"# {doc['title']}", ""]
    for heading, items in doc["sections"]:
        lines += [f"## {heading}", ""]
        for item in items:
            if isinstance(item, list):
                lines += [f"- {b}" for b in item]
            else:
                lines.append(item)
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_docx(doc: dict, path: Path) -> None:
    from docx import Document

    d = Document()
    d.add_heading(doc["title"], level=0)
    for heading, items in doc["sections"]:
        d.add_heading(heading, level=1)
        for item in items:
            if isinstance(item, list):
                for b in item:
                    d.add_paragraph(b, style="List Bullet")
            else:
                d.add_paragraph(item)
    d.save(str(path))


def write_pdf(doc: dict, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    story = [Paragraph(doc["title"], styles["Title"]), Spacer(1, 0.4 * cm)]
    for heading, items in doc["sections"]:
        story.append(Paragraph(heading, styles["Heading2"]))
        for item in items:
            if isinstance(item, list):
                bullets = [ListItem(Paragraph(b, body), leftIndent=14) for b in item]
                story.append(ListFlowable(bullets, bulletType="bullet", start="•"))
            else:
                story.append(Paragraph(item, body))
            story.append(Spacer(1, 0.2 * cm))
    SimpleDocTemplate(str(path), pagesize=A4, title=doc["title"],
                      leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writers = {".md": write_markdown, ".docx": write_docx, ".pdf": write_pdf}
    for doc in DOCS:
        path = OUT_DIR / doc["file"]
        writers[path.suffix](doc, path)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
