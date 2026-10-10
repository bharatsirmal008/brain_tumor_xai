"""Plain-language interpretation and in-memory two-stage research PDF reports."""
from io import BytesIO
from datetime import datetime, timezone
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

GRADCAM_GUIDE = (
    'The model provides a separate attention map for each MRI view. Red and yellow show stronger '
    'influence on the predicted tumor class; blue shows lower influence. Compare each overlay with '
    'its original MRI. These colors and any contour describe model attention, not a confirmed tumor '
    'boundary or segmentation. Attention can also fall on normal structures or image artifacts. '
    'The three maps are independent 2D views, not a 3D reconstruction.'
)
SHAP_GUIDE = (
    'Age and sex are compared with the model\'s reference clinical values while the MRI images stay fixed. '
    'A bar to the right of zero increases the predicted tumor-class score; a bar to the left reduces it. '
    'Longer bars mean a larger change in this internal score (logit), not a percentage. '
    'These contributions describe model behavior, not causation. A negative contribution is not evidence of No Tumor.'
)
CONFIDENCE_GUIDE = (
    'Confidence shows how strongly the classifier favors one of its four tumor classes. '
    'It is not the probability that a patient has a tumor or that the diagnosis is correct. '
    'Even a high-confidence result can be wrong; these scores are not necessarily calibrated.'
)
LIMITATIONS = (
    'Research prototype: not clinically validated. False positives and false negatives are possible. '
    'Three uploaded 2D images do not represent the complete MRI study. Training data contain '
    'source-related confounding, and performance can vary with image quality and acquisition differences. '
    'Neither heatmaps nor clinical contributions confirm disease, tumor boundaries, or tumor stage.'
)


def clinical_sentence(feature, value, predicted_class):
    direction = 'increased' if value > 0 else 'reduced' if value < 0 else 'did not change'
    return (f'{feature} {direction} the model\'s score for {predicted_class.title()} '
            f'compared with the reference clinical inputs (contribution {value:+.4f}).')


def interpretation(screening, result=None):
    if not screening['screen_positive']:
        return ('The experimental screening model did not flag a tumor in the supplied MRI views. '
                'This does not rule out a tumor or other disease. The four-class model has no No-Tumor '
                'category, so tumor classification and its explanations were skipped.')
    if result is None:
        return ('The experimental screening model flagged a possible tumor pattern. '
                'Tumor-type classification is unavailable. This screening result is not a confirmed diagnosis.')
    return (f'The screening model flagged a possible tumor pattern. Among the four tumor classes, '
            f'the second model favored {result["predicted_class"].title()} with {result["confidence"]:.2%} '
            'model confidence. A qualified clinician must interpret this alongside the complete MRI study and clinical history.')


def next_steps(screening, result=None, warnings=()):
    steps = [
        ('Share this positive screening result with a qualified clinician or radiologist for review.'
         if screening['screen_positive'] else
         'Continue the clinical review recommended by your healthcare professional. A negative screen does not rule out disease.'),
        'Review the full original MRI study and clinical history together; do not rely on these three images alone.',
        'Check image clarity, correct view placement, patient/study consistency, age, and sex before relying on the output.',
        'If confidence is low, class scores are close, or results conflict with other findings, request expert review. Confidence is not a diagnosis probability.',
        'Discuss appropriate next steps with your clinician. Do not start, stop, or change treatment based on this research output.',
    ]
    if warnings:
        steps.insert(0, 'Input quality warnings were recorded. Review and correct the inputs before interpreting the result.')
    return steps


REPORT_VERSION = "template-no-id-v1"


def build_research_report(*, saved, original_images, overlay_images):
    """Template-style assessment report without patient or study identifiers."""
    output = BytesIO()
    navy = colors.HexColor('#173F63')
    pale = colors.HexColor('#EAF2F7')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=15, leading=18, textColor=navy, alignment=1, spaceAfter=5))
    styles.add(ParagraphStyle(name='Subtitle', fontName='Helvetica', fontSize=8, leading=10, alignment=1, spaceAfter=8))
    styles['BodyText'].fontSize = 8.5
    styles['BodyText'].leading = 11
    styles['Heading2'].fontSize = 10
    styles['Heading2'].leading = 13
    styles['Heading2'].textColor = navy
    styles['Heading2'].spaceBefore = 6
    styles['Heading2'].spaceAfter = 5
    styles.add(ParagraphStyle(name='White', parent=styles['BodyText'], textColor=colors.white, fontName='Helvetica-Bold'))
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=17*mm, bottomMargin=19*mm)
    story = []
    def para(text, style='BodyText'):
        return Paragraph(escape(str(text)), styles[style])
    def add(text, style='BodyText'):
        story.append(para(text, style))
        story.append(Spacer(1, 1*mm))
    def table(rows, widths, header=False, labels=False):
        cells = [[para(c, 'White' if header and r == 0 else 'BodyText') for c in row] for r,row in enumerate(rows)]
        obj = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1 if header else 0)
        commands = [('GRID',(0,0),(-1,-1),.4,colors.HexColor('#C7D5DF')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]
        if header: commands.append(('BACKGROUND',(0,0),(-1,0),navy))
        if labels:
            for col in range(0,len(rows[0]),2): commands.append(('BACKGROUND',(col,0),(col,-1),pale))
        obj.setStyle(TableStyle(commands))
        story.append(obj)
        story.append(Spacer(1,1*mm))
    def images(data):
        cells=[]
        for view in ('axial','coronal','sagittal'):
            if view not in data:
                cells.append(para('Unavailable'))
                continue
            with PILImage.open(BytesIO(data[view])) as source:
                source=source.convert('RGB'); w,h=source.size
                buf=BytesIO();source.save(buf,format='PNG')
            buf.seek(0)
            scale=min(40*mm/w,40*mm/h)
            cells.append(Image(buf,width=w*scale,height=h*scale))
        grid=Table([cells,[para(v.title()) for v in ('axial','coronal','sagittal')]],colWidths=[58*mm]*3)
        grid.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.append(grid)
    screening=saved['screening']
    positive=screening['screen_positive']
    result=saved.get('result') if positive else None
    clinical=(result or {}).get('clinical_shapley')
    add('AI-ASSISTED BRAIN TUMOR ASSESSMENT REPORT','ReportTitle')
    add('Clinician-Facing and Patient-Friendly Explainable AI Report','Subtitle')
    add('1. Patient Information','Heading2')
    table([
        ['Date (UTC)',datetime.now(timezone.utc).strftime('%d/%m/%Y'),'Study Type','Brain MRI'],
        ['Age',f"{saved['age']} years",'Sex',saved['sex'].title()],
        ['MRI Views','Axial + Coronal + Sagittal','Use','Research only'],
    ],[28*mm,59*mm,28*mm,59*mm],labels=True)
    add('2. MRI Analysis','Heading2')
    rows=[['MRI View','AI-Assisted Analysis']]
    for view in ('axial','coronal','sagittal'):
        description = ('Original view supplied. Its model attention overlay is included in the supporting images.'
            if positive and view in overlay_images else
            'Original view supplied. Tumor-class attention was not generated because screening was negative.'
            if not positive else 'Original view supplied. Tumor-class attention is unavailable.')
        rows.append([view.title(),description])
    table(rows,[32*mm,142*mm],header=True)
    add('3. AI Prediction','Heading2')
    rows=[['Stage-1 Screening','Screen Positive' if positive else 'Screen Negative'],
          ['Screening Score / Threshold',f"{screening['tumor_probability']:.2%} / {screening['threshold']:.2%}"]]
    if result:
        rows.extend([['Predicted Tumor Type',result['predicted_class'].title()],['Prediction Confidence',f"{result['confidence']:.2%} within four tumor classes; not a diagnosis probability."]])
    else:
        rows.append(['Tumor Classification','Skipped after negative screening.' if not positive else 'Unavailable: inference did not complete.'])
    table(rows,[58*mm,116*mm],labels=True)
    add('4. Explainable AI (XAI) Findings','Heading2')
    if positive:
        cam_text=('Separate 2D heatmaps show image regions influencing the predicted class. They are not confirmed tumor boundaries or segmentation.' if overlay_images else 'Unavailable: no Grad-CAM overlays were generated.')
        shap_text=('Age and sex contributions explain changes in the predicted-class score relative to reference inputs; they are not causal effects.' if clinical else 'Unavailable: no clinical contribution values were generated.')
        table([['Method','Explanation'],['Grad-CAM',cam_text],['Shapley attribution',shap_text]],[32*mm,142*mm],header=True)
    else:
        add('Grad-CAM and clinical attribution were skipped. The four-class classifier has no No-Tumor category, so it is not run after a negative screen.')
    add('5. Natural-Language Clinical Explanation','Heading2')
    add(interpretation(screening,result))
    if clinical:
        for feature,key in [('Age','age_shap_logit'),('Sex','sex_shap_logit')]:
            add(clinical_sentence(feature,float(clinical[key]),result['predicted_class']))
    add('6. Patient-Friendly Summary','Heading2')
    add('This research result can help you discuss the model output with your doctor. Review it alongside the complete MRI study and clinical history. The model may be wrong, even when its confidence is high.' if positive else 'The model did not flag a tumor in these supplied images. This does not rule out a tumor or other disease. Continue the review recommended by your healthcare professional.')
    add('7. Clinical Disclaimer','Heading2')
    add('This AI-assisted research report is not a definitive diagnosis. The system is not clinically validated. A qualified medical professional must make diagnosis and treatment decisions using the complete clinical evaluation and appropriate tests.')
    story.append(PageBreak())
    add('SUPPORTING IMAGES AND REVIEW NOTES','ReportTitle')
    add('Original MRI Views','Heading2')
    images(original_images)
    if positive and overlay_images:
        add('Grad-CAM Overlays','Heading2');images(overlay_images)
        add('Red/yellow indicate stronger model attention; blue indicates lower attention. Colors and contours are not confirmed tumor outlines. These are independent 2D views.')
    if result:
        add('Class Probabilities','Heading2')
        table([['Class','Probability']]+[[name.title(),f'{value:.2%}'] for name,value in result['probabilities'].items()],[100*mm,74*mm],header=True)
        add('Confidence is not necessarily calibrated and is not the probability that the patient has a tumor.')
    add('Input Verification','Heading2')
    add('User confirmed the same patient/study, correct view slots, and matching age and sex. Automatic technical checks passed; they cannot establish patient identity.')
    for warning in saved.get('warnings',[]): add('Input warning: '+warning)
    if saved.get('error'): add('Execution note: '+saved['error'])
    add('Suggested Next Steps','Heading2')
    add('Share the full MRI study and this report with a qualified clinician or radiologist. Check image quality and supplied inputs. If findings conflict or confidence is low, request expert review. Do not change treatment based on this research output.')
    add('Limitations','Heading2');add(LIMITATIONS)
    def footer(canvas,document):
        canvas.setFont('Helvetica',7)
        canvas.setFillColor(colors.HexColor('#596773'))
        canvas.drawString(18*mm,15*mm,'Analysis time (UTC): '+saved['analyzed_at'])
        canvas.drawString(18*mm,11*mm,'Multimodal MRI + Clinical Data | Grad-CAM + Shapley | Research use only')
        canvas.drawRightString(192*mm,11*mm,f'Page {document.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return output.getvalue()
