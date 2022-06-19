import sys

input_commit_message = sys.argv[1]
input_extension = sys.argv[2]

#%%
from transformers import AutoTokenizer
import re
import numpy as np
import emoji
import onnxruntime as ort
from tqdm import tqdm

tqdm.pandas()

why_model_path = '../commit-quality-supplementary/tasks-inference/models/why.onnx'
imperative_model_path = '../commit-quality-supplementary/tasks-inference/models/imperative.onnx'
docs_model_path = '../commit-quality-supplementary/tasks-inference/models/docs.onnx'
bumps_model_path = '../commit-quality-supplementary/tasks-inference/models/bumps.onnx'

MODEL_NAME = 'microsoft/codebert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"additional_special_tokens": ['<pullRequestLink />', '<issueLink />', '<otherLink />']})
why_session = ort.InferenceSession(why_model_path)
imperative_session = ort.InferenceSession(imperative_model_path)
docs_session = ort.InferenceSession(docs_model_path)
bumps_session = ort.InferenceSession(bumps_model_path)

# %%
uchr = chr

_removable_emoji_components = (   # Source: https://stackoverflow.com/a/51785357
    (0x20E3, 0xFE0F),             # combining enclosing keycap, VARIATION SELECTOR-16
    range(0x1F1E6, 0x1F1FF + 1),  # regional indicator symbol letter a..regional indicator symbol letter z
    range(0x1F3FB, 0x1F3FF + 1),  # light skin tone..dark skin tone
    range(0x1F9B0, 0x1F9B3 + 1),  # red-haired..white-haired
    range(0xE0020, 0xE007F + 1),  # tag space..cancel tag
)

emoji_components = re.compile(u'({})'.format(u'|'.join([ # Source: https://stackoverflow.com/a/51785357
    re.escape(uchr(c)) for r in _removable_emoji_components for c in r])),
    flags=re.UNICODE)

def remove_emoji(text, remove_components=True):
    cleaned = emoji.replace_emoji(text, replace='')
    if remove_components:
        cleaned = emoji_components.sub(u'', cleaned)
    return cleaned

def replace_links(message):
    message = re.sub("(^.*)\(#\d+\)\n(.*)", '\\1<pullRequestLink />\n\\2', message)
    message = re.sub("(^[^\n]*)\(#\d+\)$", '\\1<pullRequestLink />', message)
    message = re.sub("pull request (#\d+)", 'pull request <pullRequestLink />', message)
    message = re.sub("#\d+", '<issueLink />', message)
    message = re.sub("https?://.*\/issues\/\d+", '<issueLink />', message)
    message = re.sub("https?://.*\/pull\/\d+", '<pullRequestLink />', message)
    message = re.sub("https?://.*\/[^ \n)]+", '<otherLink />', message)
    return message

def is_subject_separated_from_body_by_blank_line(message):
    if len(message.split('\n')) < 2:
        return True
    return message.split('\n')[1].strip() == ''

def is_subject_max_72_chars(message):
    return len(message.split('\n')[0]) <= 72

def is_subject_capitalized(message):
    message = remove_emoji(message)
    return message[0].isupper()

def subject_does_not_end_with_period(message):
    return message.split('\n')[0][-1] != '.'

def is_body_wrapped_at_72_chars(message):
    return all(list(map(lambda x: len(x) <= 72, message.split('\n'))))

def check_why(message):
  return why_session.run(None, dict(tokenizer(replace_links(message), return_tensors="np")))[0][0][0]

def is_imperative(message):
  return np.argmax(imperative_session.run(None, dict(tokenizer(replace_links(message), return_tensors="np")))) == 1

def is_documentation_change(message):
  return np.argmax(docs_session.run(None, dict(tokenizer(replace_links(message), return_tensors="np")))) == 1

def is_bump(message):
  return np.argmax(bumps_session.run(None, dict(tokenizer(replace_links(message), return_tensors="np")))) == 1

def check_beams(message):
  status = None
  score = None
  try:
    if not is_subject_separated_from_body_by_blank_line(message):
      status = 'Subject line must be separated from the commit message body by a blank line.'
      score = 'body-no-leading-empty-line'
    elif not is_subject_max_72_chars(message):
      status = 'Subject line must not be longer than 72 characters.'
      score = 'subject-longer-than-72-chars'
    elif not is_subject_capitalized(message):
      status = 'Subject line must be capitalized.'
      score = 'subject-not-capitalized'
    elif not subject_does_not_end_with_period(message):
      status = 'Subject line must not end with a period.'
      score = 'subject-ending-with-a-period'
    elif not is_body_wrapped_at_72_chars(message):
      status = 'Commit message must be wrapped at 72 characters.'
      score = 'body-not-wrapped-at-72-chars'
    elif not is_imperative(message):
      status = 'Subject line must use the imperative verb mood.'
      score = 'subject-verbs-not-imperative'
    elif is_bump(message):
      status = 'Version bump: stick to project conventions.'
      score = 'bump'
    elif is_documentation_change(message):
      status = 'Documentation change: stick to project conventions.'
      score = 'doc'
    else:
      result = check_why(message)
      if result >= 0 and result < 0.25:
        status = 'Use the body to explain what and why vs. how.'
        score = '1'
      elif result >= 0.25 and result < 0.5:
        status = 'Use the body to explain what and why vs. how.'
        score = '2'
      elif result >= 0.5 and result < 0.75:
        status = 'Commit message OK.'
        score = '3'
      elif result >= 0.75:
        status = 'Commit message OK.'
        score = '4'
    return score, status
  except:
    return 'error', 'Error'

def check_why_1_4(message):
  try:
    result = check_why(message)
    if result >= 0 and result < 0.25:
      return '1'
    elif result >= 0.25 and result < 0.5:
      return '2'
    elif result >= 0.5 and result < 0.75:
      return '3'
    elif result >= 0.75:
      return '4'
  except:
    return 'error'

print(check_beams(input_commit_message))