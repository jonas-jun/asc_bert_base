from sklearn.metrics import f1_score
from datetime import date
import pandas as pd
import numpy as np
import torch
import os

def evaluate(loader, model, criterion, opt):
    model.to(opt.device)
    model.eval()

    with torch.no_grad():
        loss, total, acc = 0.0, 0.0, 0.0
        y_pred, y_true = list(), list()
        
        for batch in loader:
            input_ids = batch['input_ids'].squeeze(1).to(opt.device)
            attention_masks = batch['attention_masks'].squeeze(1).to(opt.device)
            token_ids = batch['token_type_ids'].squeeze(1).to(opt.device)
            labels = batch['labels'].long().to(opt.device)
            if opt.pos != False:
                pos_ids = batch['pos'].squeeze(1).long().to(opt.device)
            if opt.lastid != False:
                last_ids = batch['last_ids'].to(opt.device)

            if 'bert_base' in opt.model_name:
                curloss, outputs = model(input_ids, attention_masks, token_ids, labels)
            if 'bert_attscore' in opt.model_name:
                outputs, top_k_words = model(input_ids, attention_masks, token_ids)
                curloss = criterion(outputs, labels)
            else:
                outputs = model(input_ids, attention_masks, token_ids)
                curloss = criterion(outputs, labels)

            # if (opt.pos != False) & (opt.lastid != False):
            #     outputs = model(input_ids, attention_masks, token_ids, pos_ids, last_ids)
            #     loss = criterion(outputs, labels)
            # if (opt.pos != False) & (opt.lastid == False):
            #     outputs = model(input_ids, attention_masks, token_ids, pos_ids)
            #     loss = criterion(outputs, labels)
            # if (opt.pos == False) & (opt.lastid =! False):
            #     outputs = model(input_ids, attention_masks, token_ids, last_ids)
            #     loss = criterion(outputs, labels)
            # if (opt.pos == False) & (opt.lastid == False)
            #     outputs = model(input_ids, attention_masks, token_ids)
            #     loss = criterion(outputs, labels)

            loss += curloss.item()
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            acc += torch.sum(preds==labels).item()
            total += input_ids.size(0)

    F1 = round((f1_score(y_true, y_pred, average='macro')*100), 2)
    return loss/total, acc/total, F1

def trainer(train_loader, val_loader, model, criterion, optimizer, scheduler, opt):
    model.to(opt.device)
    max_val_acc, max_val_f1, max_val_epoch, global_step = 0, 0, 0, 0

    for i_epoch in range(1, opt.num_epochs+1):
        n_correct, n_total, loss_total = 0, 0, 0
        model.train()

        for i_batch, batch in enumerate(train_loader):
            global_step += 1
            input_ids = batch['input_ids'].squeeze(1).to(opt.device) # batch, 1, length -> batch, length
            attention_masks = batch['attention_masks'].squeeze(1).to(opt.device)
            token_ids = batch['token_type_ids'].squeeze(1).to(opt.device)
            labels = batch['labels'].long().to(opt.device)

            model.zero_grad()
            
            if 'bert_base' in opt.model_name:
                loss, outputs = model(input_ids, attention_masks, token_ids, labels)
            if 'bert_attscore' in opt.model_name:
                outputs, top_k_words = model(input_ids, attention_masks, token_ids)
                loss = criterion(outputs, labels)
            else:
                outputs = model(input_ids, attention_masks, token_ids)
                loss = criterion(outputs, labels)

            # if (opt.pos != False) & (opt.lastid != False):
            #     outputs = model(input_ids, attention_masks, token_ids, pos_ids, last_ids)
            #     loss = criterion(outputs, labels)
            # if (opt.pos != False) & (opt.lastid == False):
            #     outputs = model(input_ids, attention_masks, token_ids, pos_ids)
            #     loss = criterion(outputs, labels)
            # if (opt.pos == False) & (opt.lastid =! False):
            #     outputs = model(input_ids, attention_masks, token_ids, last_ids)
            #     loss = criterion(outputs, labels)
            # if (opt.pos == False) & (opt.lastid == False)
            #     outputs = model(input_ids, attention_masks, token_ids)
            #     loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            n_correct += (torch.argmax(outputs, -1)==labels).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)
            
            if global_step % opt.log_step == 0:
                train_loss = loss_total / n_total
                train_acc = n_correct / n_total
                print('   global step: {:,} | train loss: {:.3f}, train_acc: {:.2f}%'\
                      .format(global_step, train_loss, train_acc*100))
            
        val_loss, val_acc, val_f1 = evaluate(val_loader, model, criterion, opt)
        
        if i_epoch >= 3:
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{}_{}_{}_epoch_{}_val_acc_{}%'.format(opt.model_name, opt.dataset_series, 
                    opt.dataset_domain, i_epoch, round(val_acc*100, 2))
                #path = os.path.join('content/drive/MyDrive/research', path) # for colab state_dict folder
                torch.save(model.state_dict(), path)
                print('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= opt.patience:
                print('>> early stop')
                break

        print('Epoch: {:02d} | Val Loss: {:.3f} | Val Acc: {:.2f}%'.format(i_epoch, val_loss, val_acc*100))
    
    print('Best Val Acc: {:.2f}% at {} epoch'.format(max_val_acc*100, max_val_epoch))
    best_path = 'state_dict/BEST_{}_{}_{}_val_acc_{}%'.format(opt.model_name, opt.dataset_series,
        opt.dataset_domain, round(max_val_acc*100, 2))
    #best_path = os.path.join('/content/drive/MyDrive/research', best_path) # for colab state_dict folder
    torch.save(model.state_dict(), best_path)
    print('>> saved best state dict: {}'.format(best_path))
    return max_val_acc, max_val_epoch, best_path

def runs(trainer, train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, opt):
    # keep first state
    if not os.path.exists('state_dict'):
        os.mkdir('state_dict')
    torch.save(model.state_dict(), 'state_dict/start_state')

    result_dict = dict()
    result_dict['model_name'] = list()
    result_dict['task'] = list()
    result_dict['dataset'] = list()
    result_dict['run'] = list()
    result_dict['max_val_acc'] = list()
    result_dict['max_val_epoch'] = list()
    result_dict['best_path'] = list()
    result_dict['date'] = list()
    result_dict['test_loss'] = list()
    result_dict['test_acc'] = list()
    result_dict['test_f1'] = list()

    model_name = opt.model_name
    task = 'sub: {}, task: {}'.format(opt.subtask, opt.task)
    dataset = 'series: {}, domain: {}'.format(opt.dataset_series, opt.dataset_domain)
    today_str = str(date.today())
    
    for run in range(1, opt.runs+1):
        print('>>>>> RUN NUMBER: {:02d} <<<<<'.format(run))
        max_val_acc, max_val_epoch, path = trainer(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion,
                                                   optimizer=optimizer, scheduler=scheduler, opt=opt)
        
        model.load_state_dict(torch.load(path))
        test_loss, test_acc, test_f1 = evaluate(loader=test_loader, model=model, criterion=criterion, opt=opt)
        print('RUN: {:02d} | Test loss: {:.3f} | Test_acc: {:.2f}% | Test_f1: {:.2f}'.format(run, test_loss, test_acc*100, test_f1))
        
        result_dict['model_name'].append(model_name)
        result_dict['task'].append(task)
        result_dict['dataset'].append(dataset)
        result_dict['run'].append(run)
        result_dict['max_val_acc'].append(max_val_acc)
        result_dict['max_val_epoch'].append(max_val_epoch)
        result_dict['best_path'].append(path)
        result_dict['date'].append(today_str)
        result_dict['test_loss'].append(test_loss)
        result_dict['test_acc'].append(test_acc)
        result_dict['test_f1'].append(test_f1)

        print('>>>>> RUN {:02d} HAS BEEN FINISHED <<<<<'.format(run))
        model.load_state_dict(torch.load('state_dict/start_state'))       
    
    #best test acc를 찾아서 result_dict['test_acc']에서 index를 구한 다음 그 path를 반환!
    best_test_acc = np.max(result_dict['test_acc'])
    best_test_acc_idx = np.where(result_dict['test_acc']==best_test_acc)[0][-1]
    best_run_path = result_dict['best_path'][best_test_acc_idx]
    
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv('state_dict/[RUN_RESULT]{}_{}_{}_{}.csv'.format(model_name, opt.task, opt.dataset_series, opt.dataset_domain))
    export_multi_sheets(df=result_df, filename='state_dict/[RUN_RESULT]{}_{}_{}_{}.xlsx'.format(model_name, opt.task, opt.dataset_series, opt.dataset_domain))
    print('='*20)
    print('best run: {:02d} | best acc: {:.2f} | best f1: {:.2f} | best run path: {}'
        .format(best_test_acc_idx+1, best_test_acc*100, result_dict['test_f1'][best_test_acc_idx], best_run_path))
    print('<<<Averages>>>')
    mean_loss = round(np.mean(result_dict['test_loss']), 3)
    mean_acc = round(np.mean(result_dict['test_acc']), 4)
    mean_f1 = round(np.mean(result_dict['test_f1']), 2)
    print('test_loss: {:.3f} | test_acc: {:.2f} | test_f1: {:.2f}'.format(mean_loss, mean_acc*100, mean_f1))
    return result_dict, best_run_path

def export_multi_sheets(df, filename):
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, index=False, sheet_name='result')
        df.describe().to_excel(writer, index=True, sheet_name='describe')
    print('Sheets has been exported')



# for insert mode