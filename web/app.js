'use strict';
const $ = id => document.getElementById(id);
let currentReport = null;
const text = $('text-input');
document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-selected', String(b === button)); });
  document.querySelectorAll('.panel').forEach(p => { p.hidden = p.id !== button.dataset.panel; });
  $('result').hidden = true; $('error').hidden = true;
}));
text.addEventListener('input', () => { $('count').textContent = `${Array.from(text.value).length} символов`; });
function metric(value, label) {
  const box = document.createElement('div'); box.className = 'metric';
  const strong = document.createElement('strong'); strong.textContent = value;
  const caption = document.createElement('span'); caption.textContent = label;
  box.append(strong, caption); $('metrics').append(box);
}
function show(result) {
  currentReport = result;
  $('result').hidden = false; $('metrics').replaceChildren(); $('unicode-list').replaceChildren(); $('trace').replaceChildren();
  const ordinary = result.status === 'vendor_verification_unavailable';
  $('trace-section').hidden = ordinary || !result.token_trace?.length;
  $('unicode-section').hidden = !ordinary;
  if (ordinary) {
    $('result-title').textContent = 'Метка Claude: вывод недоступен';
    $('result-summary').textContent = 'Локальная проверка выполнена. Определить наличие метки Claude без её профиля или официального детектора нельзя.';
    metric(result.characters, 'символов'); metric(result.words_approx, 'слов, приблизительно');
    metric(Object.values(result.unicode_findings).reduce((a,b) => a+b,0), 'специальных Unicode-символов');
    const entries = Object.entries(result.unicode_findings);
    if (!entries.length) { const li = document.createElement('li'); li.textContent = 'Проверяемые специальные символы не найдены.'; $('unicode-list').append(li); }
    entries.forEach(([name, count]) => { const li = document.createElement('li'); li.textContent = `${name}: ${count}`; $('unicode-list').append(li); });
    $('result-limit').textContent = 'Наличие или отсутствие этих символов не определяет авторство. Текст не изменён, не сохранён на диск и не отправлен внешнему сервису.';
  } else {
    const titles = {reference_signal_detected:'Сигнал известного профиля обнаружен',reference_signal_not_detected:'Сигнал этого профиля не обнаружен',uncalibrated:'Балл рассчитан — нужна калибровка',insufficient_contexts:'Недостаточно учитываемых контекстов'};
    $('result-title').textContent = titles[result.status] || 'Заключение недоступно';
    $('result-summary').textContent = result.synthetic_example ? 'Контрольный эксперимент на искусственных токенах. Результат относится только к публичным ключам этого теста.' : 'Результат относится только к предоставленному профилю и указанной контрольной выборке. Происхождение из Claude не проверено.';
    metric(result.mean_g_score === null ? '—' : result.mean_g_score.toFixed(4), 'средний g-балл; не вероятность авторства');
    metric(result.scored_contexts, 'учтённых контекстов');
    metric(result.empirical_tail_rank === null ? '—' : result.empirical_tail_rank.toFixed(4), 'доля в хвосте контрольной выборки, с поправкой +1');
    (result.token_trace || []).forEach(row => { const bar = document.createElement('span'); bar.className = `bar${row.included ? '' : ' excluded'}`; bar.style.height = `${Math.max(3,row.g_mean*100)}%`; bar.title = `Позиция ${row.position}: токен ${row.token_id}, g=${row.g_mean.toFixed(3)}${row.included?'':' — исключён'}`; $('trace').append(bar); });
    $('result-limit').textContent = result.empirical_tail_rank === null ? 'Нет сопоставимой калибровки либо слишком мало контекстов. Балл сам по себе не даёт заключения. Проверка Claude недоступна.' : `Порог 0,01; контрольных примеров: ${result.calibration_samples}. Калибровка применима только к указанному источнику данных и одиночному тесту. Чужой ключ, другой токенизатор или иной текст могут изменить результат. Проверка Claude недоступна.`;
  }
}
async function request(path, payload) {
  $('error').hidden = true;
  const buttons = [...document.querySelectorAll('button')]; buttons.forEach(b => b.disabled = true);
  try {
    const response = await fetch(path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Ошибка проверки'); show(result);
  } catch (error) { $('error').textContent = error.message; $('error').hidden = false; }
  finally { buttons.forEach(b => b.disabled = false); }
}
$('analyze').addEventListener('click', () => request('/api/analyze', {text:text.value}));
document.querySelectorAll('[data-demo]').forEach(button => button.addEventListener('click', () => request('/api/demo', {kind:button.dataset.demo})));
async function readFile(event, tokens) {
  const file = event.target.files[0]; if (!file) return;
  try {
    if (file.size > 512000) throw new Error('Файл должен быть меньше 512 КБ.');
    const bytes = await file.arrayBuffer(); const raw = new TextDecoder('utf-8',{fatal:true}).decode(bytes);
    if (tokens) { const payload = JSON.parse(raw); await request('/api/tokens', payload.payload || payload); }
    else { if (Array.from(raw).length > 100000) throw new Error('Не более 100 000 символов.'); text.value = raw; text.dispatchEvent(new Event('input')); }
  } catch (error) { $('error').textContent = error.message; $('error').hidden = false; }
  event.target.value = '';
}
$('text-file').addEventListener('change', event => readFile(event, false));
$('token-file').addEventListener('change', event => readFile(event, true));
$('export').addEventListener('click', () => {
  if (!currentReport) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(currentReport,null,2)],{type:'application/json'}));
  const anchor = document.createElement('a'); anchor.href=url; anchor.download='watermark-report.json'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});
